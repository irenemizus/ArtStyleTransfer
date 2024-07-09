import copy
import os
import traceback

import cv2

MYPATH = os.path.dirname(os.path.abspath(__file__))
print(f"Script's directory: {MYPATH}")
os.environ["TORCH_HOME"] = MYPATH

import math_utils
from torchvision import transforms

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import argparse
import asyncio

# ImageNet statistics
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


class ContentStylePair:
    """ Pairs content image - style image """
    def __init__(self, content, style):
        self.content = content      # (content_img_name, content_img)
        self.style = style          # (style_img_name, style_img)


class RepresentationBuilder:
    """ A class for calculating of input (content or style) image representations by a set of the given neural net's feature maps """
    def __init__(self, image, neural_net):
        self.__image = image
        self.__neural_net = neural_net

        self.__features = neural_net(image)

    def build_content(self, feature_map_indices: int | list[int]):
        list_taken = isinstance(feature_map_indices, list)
        indices = feature_map_indices if list_taken else [ feature_map_indices ]

        # content representations
        rep = [ x.squeeze(axis=0) for index, x in enumerate(self.__features) if index in indices ]

        return rep if list_taken else rep[0]

    def build_style(self, feature_map_indices: int | list[int]):
        list_taken = isinstance(feature_map_indices, list)
        indices = feature_map_indices if list_taken else [ feature_map_indices ]

        # style representations
        rep = [ math_utils.gram_matrix(x) for index, x in enumerate(self.__features) if index in indices ]

        return rep if list_taken else rep[0]


class LossBuilder:
    """ A class for calculating loss functions """
    def __init__(self, content_feature_maps_index, style_feature_maps_indices, target_content_image, target_style_image, neural_net, content_weight, style_weight, tv_weight):
        self.__content_feature_maps_index = content_feature_maps_index
        self.__style_feature_maps_indices = style_feature_maps_indices
        self.__target_content_image = target_content_image
        self.__target_style_image = target_style_image
        self.__neural_net = neural_net
        self.__content_weight = content_weight
        self.__style_weight = style_weight
        self.__tv_weight = tv_weight

        content_rep_builder = RepresentationBuilder(image=target_content_image, neural_net=neural_net)
        style_rep_builder = RepresentationBuilder(image=target_style_image, neural_net=neural_net)

        self.__target_content_representation = content_rep_builder.build_content(content_feature_maps_index)
        self.__target_style_representation = style_rep_builder.build_style(style_feature_maps_indices)

    def build(self, optimizing_img):
        current_rep_builder = RepresentationBuilder(image=optimizing_img, neural_net=self.__neural_net)

        current_content_representation = current_rep_builder.build_content(self.__content_feature_maps_index)
        # discrepancy from the content image
        content_loss = torch.nn.MSELoss(reduction='mean')(self.__target_content_representation, current_content_representation)

        current_style_representation = current_rep_builder.build_style(self.__style_feature_maps_indices)

        # calculating style representation for hires
        style_loss = 0.0
        for gram_gt, gram_hat in zip(self.__target_style_representation, current_style_representation):
            # discrepancy from the style image
            style_loss += torch.nn.MSELoss(reduction='mean')(gram_gt[0], gram_hat[0])
        style_loss /= len(self.__target_style_representation)

        # total variation loss (rough denoiser)
        tv_loss = math_utils.total_variation(optimizing_img) # + utils.regularization(optimizing_img)

        # total loss
        total_loss = self.__content_weight * content_loss + self.__style_weight * style_loss + self.__tv_weight * tv_loss

        return total_loss, content_loss, style_loss, tv_loss


class NeuralStyleTransfer:
    """ The main class for calculating of artistic style transfer """
    def __init__(self, device, model_name, style_imgs, optimizer_name):
        self.__device = device
        self.__model_name = model_name
        self.__style_imgs = style_imgs
        self.__optimizer_name = optimizer_name

    async def process(self, content_imgs, init_img, lr_start, iters_num, content_weight, style_weight, tv_weight, init_img_name):
        # obtaining feature map indices
        neural_net, content_feature_maps_index, style_feature_maps_indices = math_utils.prepare_model(self.__model_name, self.__device)
        print(f'Using {self.__model_name} in the optimization procedure.')

        init_img = prepare_img(init_img, self.__device)
        # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
        optimizing_img = Variable(init_img, requires_grad=True)

        if self.__optimizer_name == 'adam':
            optimizer = Adam((optimizing_img,), lr=lr_start)
        elif self.__optimizer_name == 'lbfgs':
            optimizer = LBFGS((optimizing_img,), max_iter=1, line_search_fn='strong_wolfe', lr=lr_start)
        else:
            raise RuntimeError("Unknown optimizer")

        loss_builders = []
        for content_img, style_img in zip(content_imgs, self.__style_imgs):
            content_img = prepare_img(content_img, self.__device)
            style_img = prepare_img(style_img, self.__device)

            loss_builders.append(LossBuilder(content_feature_maps_index, style_feature_maps_indices, content_img, style_img,
                                            neural_net, content_weight, style_weight, tv_weight))

        step = 0
        torch.autograd.set_detect_anomaly(True)

        def optimizer_step_callback():
            try:
                lr = 0
                for g in optimizer.param_groups:
                    g['lr'] *= 0.999
                    lr = g['lr']
                print(f"new lr = {lr}")

                nonlocal step
                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                # Building lower resolutions for optimizing_img
                print(f'{self.__optimizer_name} | processing image: {init_img_name} | iteration: {step:03} :')
                optimizing_img_levels = None
                total_loss = None
                for i in range(len(loss_builders)):
                    if i == 0:
                        optimizing_img_levels = [ optimizing_img ]
                    else:
                        sw = optimizing_img_levels[i - 1].shape[2]
                        sh = optimizing_img_levels[i - 1].shape[3]
                        conv_twice = torch.nn.functional.interpolate(optimizing_img_levels[i - 1], size=(sw // 2, sh // 2), mode='bicubic')
                        optimizing_img_levels.append(conv_twice)

                    total_loss_l, content_loss, style_loss, tv_loss = loss_builders[i].build(optimizing_img_levels[i])
                    if total_loss is None:
                        total_loss = total_loss_l
                    else:
                        previous_loss_importance = 1.0
                        total_loss = previous_loss_importance * total_loss + total_loss_l
                        #total_loss = torch.pow(total_loss * total_loss_l, 0.5)

                    with torch.no_grad():
                        print(f' - level {i} | level loss={total_loss_l.item():.3e}, content_loss={content_weight * content_loss.item():.3e}, style loss={style_weight * style_loss:.3e}, tv loss={tv_weight * tv_loss.item():.3e}')

                if total_loss.requires_grad:
                    total_loss.backward()

                with torch.no_grad():
                    print(f'{self.__optimizer_name} | total loss={total_loss.item():.3e}')

                step += 1
                return total_loss
            except:
                traceback.print_exc()
                raise

        while step < iters_num:
            await asyncio.get_running_loop().run_in_executor(None, optimizer.step, optimizer_step_callback)
            res_img = copy.deepcopy(optimizing_img)
            yield unprepare_img(res_img), step


async def resize(img, level):
    base_diameter = 256

    current_height, current_width = img.shape[:2]
    if current_height >= current_width:
        base_width = base_diameter
        base_height = int(base_width * (current_height / current_width))
    else:
        base_height = base_diameter
        base_width = int(base_height * (current_width / current_height))

    new_width = base_width * pow(2, level)
    new_height = base_height * pow(2, level)

    return cv2.resize(copy.deepcopy(img), (new_width, new_height), interpolation=cv2.INTER_CUBIC)


GLOBAL_VALUE_DON_T_USE = 0


async def neural_style_transfer(content_n_style: ContentStylePair,
                                content_weight, style_weight, tv_weight,
                                optimizer, model, init_method,
                                iters_num, levels_num, noise_factor, noise_levels, noise_levels_central_amplitude,
                                noise_levels_peripheral_amplitude, noise_levels_dispersion):
    print("entering neural_style_transfer")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"

    #if device == "cuda":
    #    global GLOBAL_VALUE_DON_T_USE
    #    coin = GLOBAL_VALUE_DON_T_USE
    #    GLOBAL_VALUE_DON_T_USE = (GLOBAL_VALUE_DON_T_USE + 1) % 2
    #    device = f"{device}:{coin}"

    device =  torch.device(device)

    model_name = model
    optimizer_name = optimizer

    level = 0

    content_img_level0 = await resize(content_n_style.content[1], level=level)
    style_img_level0 = await resize(content_n_style.style[1], level=level)

    # Starting the processing
    content_img_levels = [ content_img_level0 ]
    style_img_levels = [ style_img_level0 ]
    print("entering levels")

    for level in range(1, levels_num):

        content_img_next = await resize(content_n_style.content[1], level=level)
        style_img_next = await resize(content_n_style.style[1], level=level)

        content_img_levels.insert(0, content_img_next)
        style_img_levels.insert(0, style_img_next)


    # Making blurry noise with big granularity
    noise_shape = content_img_levels[0].shape
    nw = noise_shape[1]
    nh = noise_shape[0]
    gaussian_noise_img = np.zeros(noise_shape, dtype=np.float32)
    for noise_granularity, central_amplitude, peripheral_amplitude, dispersion_scale in zip(noise_levels, noise_levels_central_amplitude, noise_levels_peripheral_amplitude, noise_levels_dispersion):
        if noise_granularity == 0:
            # Granularity zero means constant value, no noise.
            constant_level_gauss_mask = gaussian_mask(noise_shape, central_amplitude, peripheral_amplitude, dispersion_scale)
            gaussian_noise_img += constant_level_gauss_mask
        else:
            if noise_granularity > 0:
                # Positive granularity value means the number of noise spots along the shortest image axis
                # The noise spot size in this case is proportional to the image size
                noise_spots_number = noise_granularity

                if nh <= nw:
                    noise_shape_div_h = noise_spots_number
                    noise_shape_div_w = nw * noise_spots_number // nh
                else:
                    noise_shape_div_w = noise_spots_number
                    noise_shape_div_h = nh * noise_spots_number // nw
            else:
                # Negative noise granularity means the size of a spot (i.e. -2 means spot sized 2x2 on every resolution)
                noise_shape_div_w = nw // (-noise_granularity)
                noise_shape_div_h = nh // (-noise_granularity)

            noise_shape_div = (noise_shape_div_h, noise_shape_div_w, noise_shape[2])

            USE_NORMAL_NOISE_JUST_FOR_DEMONSTRATION = False
            if USE_NORMAL_NOISE_JUST_FOR_DEMONSTRATION:
                gaussian_noise_img_lowres = np.clip(np.random.normal(loc=0, scale=255, size=noise_shape_div).astype(np.float32) / 255, 0.0, 1.0)
            else:
                gaussian_noise_img_lowres = make_style_noise(style_img_levels[0], noise_shape_div)

            gaussian_noise_img_level = cv2.resize(gaussian_noise_img_lowres, dsize=(nw, nh),
                                                  interpolation=cv2.INTER_CUBIC)
            noise_level_gauss_mask = gaussian_mask(gaussian_noise_img_level.shape, central_amplitude, peripheral_amplitude, dispersion_scale)
            gaussian_noise_img += gaussian_noise_img_level * noise_level_gauss_mask


    temp_img = copy.deepcopy(gaussian_noise_img)
    #temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("noise_mask.jpg", temp_img * 255)
    temp_img = cv2.GaussianBlur(temp_img, (107, 107), 0)
    cv2.imwrite("noise_mask_blurry.jpg", temp_img * 255)
    #gaussian_noise_img /= sum(noise_level_powers)

    IGNORE_GRADIENT_MAP_JUST_FOR_DEMONSTRATION = False
    if IGNORE_GRADIENT_MAP_JUST_FOR_DEMONSTRATION:
        noise_replacement = noise_factor
    else:
        # Compute gradients along the X and Y axis, respectively
        sobelX = cv2.Sobel(content_img_levels[0], cv2.CV_64F, 1, 0, ksize=5)
        sobelY = cv2.Sobel(content_img_levels[0], ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
        sobelX = np.absolute(sobelX)
        sobelY = np.absolute(sobelY)

        sobelCombined = np.sqrt(sobelX*sobelX + sobelY*sobelY)
        sobelCombined = np.clip(sobelCombined, 0.0, 100)

        sobelCombined = cv2.GaussianBlur(sobelCombined, ksize=(101, 101), sigmaX=0.2)
        a = 5.0
        noise_replacement = a * noise_factor / (a + sobelCombined)

        cv2.imwrite("test_noise_rep.jpg", noise_replacement * 255)
        cv2.imwrite("test_noise_rep_blurry.jpg", noise_replacement * 255)


    if init_method == 'random':
        init_img_next = gaussian_noise_img * 0.5
        init_img_name = 'random'
    elif init_method == 'content+noise':
        init_img_next = await resize(content_n_style.content[1], level=level)
        init_img_name = content_n_style.content[0]

        init_img_next = ((1.0 - noise_replacement) * init_img_next + noise_replacement * gaussian_noise_img).astype(np.float32)
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style_img_resized = await resize(content_n_style.style[1], level=level)
        init_img_next = style_img_resized
        init_img_name = content_n_style.style[0]

    nst = NeuralStyleTransfer(device, model_name, style_img_levels, optimizer_name)
    print("entering processing loop")
    lr_start = 10.0
    async for img, cur_iter in nst.process(content_img_levels, init_img_next, lr_start, iters_num, content_weight, style_weight, tv_weight, init_img_name):
        print("in the processing loop")
        percent = cur_iter / iters_num * 100.0
        cur_iter += 1
        yield percent, img





def prepare_img(img, device):
    # normalize using ImageNet's mean
    # [0, 255] range worked much better for me than [0, 1] range (even though PyTorch models were trained on latter)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    return transform(img.copy()).to(device).unsqueeze(0)


def unprepare_img(img):
    # reversing prepare_img()
    dump_img = img.permute([0, 2, 3, 1]).squeeze(0).to("cpu").detach().numpy()
    dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))

    dump_img = dump_img.astype(np.float32) / 255

    return dump_img


def gaussian_mask(shape, central_amplitude, peripheral_amplitude, dispersion_scale=0.5):    # Extracting the height and width of an image
    rows, cols = shape[:2]
    nw = shape[1]
    nh = shape[0]

    # generating vignette mask using Gaussian
    # resultant_kernels
    X_resultant_kernel = cv2.getGaussianKernel(cols, nw * dispersion_scale)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, nh * dispersion_scale)

    # generating resultant_kernel matrix
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T

    gauss_norm = resultant_kernel / resultant_kernel[rows // 2, cols // 2]

    # The gaussian mask functtion shall rise from central_amplitude in the centter to (almost) 1.0 on the edges
    mask = peripheral_amplitude + gauss_norm * (central_amplitude - peripheral_amplitude) #/ np.linalg.norm(resultant_kernel)

    #cv2.imwrite("test_mask.jpg", mask * 255)

    # applying the mask to each channel in the input image
    #for i in range(3):
    #    image[:, :, i] = image[:, :, i] * mask

    expanded = np.expand_dims(mask, 2)
    res = np.repeat(expanded, 3, axis=2)
    return res



def make_style_noise(style_img_np, targ_shape):
    #cv2.imwrite("test_style.jpg", style_img_np * 255)
    nw = targ_shape[1]
    nh = targ_shape[0]
    inp_img_copy = copy.deepcopy(style_img_np)
    style_img_np_resized = cv2.resize(inp_img_copy, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)

    style_vect = style_img_np_resized.reshape(nh * nw, -1)  #.flatten(style_img_np_resized)
    style_noise_vect = np.random.permutation(style_vect)

    res = style_noise_vect.reshape(targ_shape)

    # res[:, :, 0] = (res[:,:,0] + res[:,:,1] + res[:,:,2]) / 3
    # res[:, :, 1] = res[:,:,0]
    # res[:, :, 2] = res[:,:,0]

    #cv2.imwrite("test.jpg", res * 255)
    return res


# if __name__ == "__main__":
#     #
#     # fixed args - don't change these unless you have a good reason
#     #
#     default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
#     content_images_dir = os.path.join(default_resource_dir, 'content-images')
#     style_images_dir = os.path.join(default_resource_dir, 'style-images')
#     #
#     # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
#     # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
#     #
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--content_img_name", type=str, help="content image name", default='figures.jpg')
#     parser.add_argument("--style_img_name", type=str, help="style image name", default='vg_starry_night.jpg')
#
#     parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e1)
#     parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=1e5)
#     parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=0e3)
#     parser.add_argument("--noise_factor", type=float, help="strength of noise, which is added to the initial image", default=0.95)
#     parser.add_argument("--noise_levels", type=tuple, help="tuple of noise spots number along the shortest axis of the content image for a few noise levels", default=(32, 16, 8, 1, 0))
#     parser.add_argument("--noise_levels_central_amplitude", type=tuple,
#                         help="tuple of noise envelope strength in the center of the image for a few noise levels",
#                         default=(0.30, 0.20, 0.10, 0.20, 0.20))
#     parser.add_argument("--noise_levels_peripheral_amplitude", type=tuple,
#                         help="tuple of noise envelope strength at the periphery of the image for a few noise levels",
#                         default=(0.20, 0.30, 0.40, 0.10, 0.00))
#     parser.add_argument("--noise_levels_dispersion", type=tuple,
#                         help="tuple of noise envelope dispersion for a few noise levels",
#                         default=(0.20, 0.30, 0.40, 0.60, 0.30))
#
#     parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
#     parser.add_argument("--model", type=str, choices=['vgg19'], default='vgg19')
#     parser.add_argument("--init_method", type=str, choices=['random', 'content+noise', 'style'], default='content+noise')
#     parser.add_argument("--iters_num", type=int, help="number of iterations to perform", default=200)
#     parser.add_argument("--levels_num", type=int, help="number of pyramid levels", default=3)
#
#     args = parser.parse_args()
#
#     content_img_name = args.content_img_name
#     style_img_name = args.style_img_name
#     content_weight = args.content_weight
#     style_weight = args.style_weight
#     tv_weight = args.tv_weightn
#     noise_factor = args.noise_factor
#     noise_levels = args.noise_levels
#     noise_levels_central_amplitude = args.noise_levels_central_amplitude
#     noise_levels_peripheral_amplitude = args.noise_levels_peripheral_amplitude
#     noise_levels_dispersion = args.noise_levels_dispersion
#     optimizer = args.optimizer
#     model = args.model
#     init_method = args.init_method
#     iters_num = args.iters_num
#     levels_num = args.levels_num
#
#     # original NST (Neural Style Transfer) algorithm (Gatys et al.)
#     content_img_path = os.path.join(content_images_dir, content_img_name)
#     style_img_path = os.path.join(style_images_dir, style_img_name)
#
#     content_n_style = ContentStylePair((content_img_name, lab.load_image(content_img_path)), (style_img_name, lab.load_image(style_img_path)))
#     results_path = neural_style_transfer(content_n_style, content_weight, style_weight, tv_weight, optimizer, model,
#                                          init_method, iters_num, levels_num,
#                                          noise_factor, noise_levels, noise_levels_central_amplitude,
#                                          noise_levels_peripheral_amplitude, noise_levels_dispersion)