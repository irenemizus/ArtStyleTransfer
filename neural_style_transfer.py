import copy
import os
import traceback

import cv2
from torch import Tensor

MYPATH = os.path.dirname(os.path.abspath(__file__))
print(f"Script's directory: {MYPATH}")
os.environ["TORCH_HOME"] = MYPATH

import math_utils
from torchvision import transforms

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import asyncio

# ImageNet statistics
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

# Flags for debug/demonstration
USE_NORMAL_NOISE_JUST_FOR_DEMONSTRATION = False
WITHOUT_GAUSSIAN_MASK_JUST_FOR_DEMONSTRATION = False
SHOW_TEST_IMGS = False
IGNORE_GRADIENT_MAP_JUST_FOR_DEMONSTRATION = False


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

        # Adding random noise on each step (experimental (not documented) feature)
        noise_power = 0.2

        content_noise = torch.clip((0.5 * torch.randn(self.__target_content_representation.shape)) + 0.5, min=0.0, max=1.0)
        content_noise = content_noise.to(self.__target_content_representation.device)
        noised_target = torch.pow(self.__target_content_representation, (1.0 - noise_power)) * torch.pow(content_noise, noise_power)

        content_loss = torch.nn.MSELoss(reduction='mean')(noised_target, current_content_representation)

        current_style_representation = current_rep_builder.build_style(self.__style_feature_maps_indices)

        style_noise = [torch.clip((0.5 * torch.randn(self.__target_style_representation[ind].shape)) + 0.5, min=0.0, max=1.0) for ind in range(len(self.__target_style_representation))]
        style_noise = [style_noise[ind].to(self.__target_style_representation[ind].device) for ind in range(len(style_noise))]

        noised_style = [torch.pow(self.__target_style_representation[ind], (1.0 - noise_power)) * torch.pow(style_noise[ind], noise_power) for ind in range(len(style_noise))]


        # calculating style representation for hires
        style_loss = 0.0
        for gram_gt, gram_hat in zip(noised_style, current_style_representation):
            # discrepancy from the style image
            style_loss += torch.nn.MSELoss(reduction='mean')(gram_gt[0], gram_hat[0])
        style_loss /= len(noised_style)

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

        # choosing optimizer type
        if self.__optimizer_name == 'adam':
            optimizer = Adam((optimizing_img,), lr=lr_start)
        elif self.__optimizer_name == 'lbfgs':
            optimizer = LBFGS((optimizing_img,), max_iter=1, line_search_fn='strong_wolfe', lr=lr_start)
        else:
            raise RuntimeError("Unknown optimizer")

        # calculating losses
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
                # learning rate schedule
                lr = 0
                for g in optimizer.param_groups:
                    g['lr'] *= 0.999
                    lr = g['lr']
                print(f"new lr = {lr}")

                nonlocal step
                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                print(f'{self.__optimizer_name} | processing image: {init_img_name} | iteration: {step:03} :')
                optimizing_img_levels = None
                total_loss = None
                for i in range(len(loss_builders)):
                    # Building lower resolutions for optimizing_img
                    if i == 0:
                        optimizing_img_levels = [ optimizing_img ]
                    else:
                        sw = optimizing_img_levels[i - 1].shape[2]
                        sh = optimizing_img_levels[i - 1].shape[3]
                        conv_twice = torch.nn.functional.interpolate(optimizing_img_levels[i - 1], size=(sw // 2, sh // 2), mode='bicubic')
                        optimizing_img_levels.append(conv_twice)

                    # calculating losses for the current pyramid level
                    total_loss_l, content_loss, style_loss, tv_loss = loss_builders[i].build(optimizing_img_levels[i])
                    if total_loss is None:
                        total_loss = total_loss_l
                    else:
                        # accumulating total loss from the whole pyramid
                        previous_loss_importance = 1.0
                        total_loss = previous_loss_importance * total_loss + total_loss_l
                        #total_loss = torch.pow(total_loss * total_loss_l, 0.5)

                    with torch.no_grad():
                        print(f' - level {i} | level loss={total_loss_l.item():.3e}, content_loss={content_weight * content_loss.item():.3e}, style loss={style_weight * style_loss:.3e}, tv loss={tv_weight * tv_loss.item():.3e}')

                # backward propagation
                if total_loss.requires_grad:
                    total_loss.backward()

                with torch.no_grad():
                    print(f'{self.__optimizer_name} | total loss={total_loss.item():.3e}')

                step += 1
                return total_loss
            except:
                traceback.print_exc()
                raise

        # the main optimization loop
        while step < iters_num:
            await asyncio.get_running_loop().run_in_executor(None, optimizer.step, optimizer_step_callback)
            res_img = copy.deepcopy(optimizing_img)
            yield unprepare_img(res_img), step


async def resize(img, level):
    """ A function for proper resizing of an image according to the level of pyramid """
    base_diameter = 256 # zero pyramid level

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


async def neural_style_transfer(content_n_style: ContentStylePair,
                                content_weight, style_weight, tv_weight,
                                optimizer, model, init_method,
                                iters_num, levels_num, noise_factor, noise_levels, noise_levels_central_amplitude,
                                noise_levels_peripheral_amplitude, noise_levels_dispersion):
    """ The main function """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #GLOBAL_VALUE_DON_T_USE = 0
    #if device == "cuda":
    #    global GLOBAL_VALUE_DON_T_USE
    #    coin = GLOBAL_VALUE_DON_T_USE
    #    GLOBAL_VALUE_DON_T_USE = (GLOBAL_VALUE_DON_T_USE + 1) % 2
    #    device = f"{device}:{coin}"

    device =  torch.device(device)
    model_name = model
    optimizer_name = optimizer

    # Building a pyramid for content ang style images
    level = 0
    content_img_level0 = await resize(content_n_style.content[1], level=level)
    style_img_level0 = await resize(content_n_style.style[1], level=level)

    # Starting the processing
    content_img_levels = [ content_img_level0 ]
    style_img_levels = [ style_img_level0 ]

    for level in range(1, levels_num):
        content_img_next = await resize(content_n_style.content[1], level=level)
        style_img_next = await resize(content_n_style.style[1], level=level)

        content_img_levels.insert(0, content_img_next)
        style_img_levels.insert(0, style_img_next)

    # Making a style-based noise with different levels of granularity and gaussian envelopes
    noise_shape = content_img_levels[0].shape
    nw = noise_shape[1]
    nh = noise_shape[0]
    # noise map
    gaussian_noise_img = np.zeros(noise_shape, dtype=np.float32)
    for noise_granularity, central_amplitude, peripheral_amplitude, dispersion_scale in zip(noise_levels, noise_levels_central_amplitude, noise_levels_peripheral_amplitude, noise_levels_dispersion):
        if noise_granularity == 0:
            # Granularity zero means constant value, no noise (it is needed for keeping the same brightness level for the whole noise map)
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

            # shape of the low-resolution noise map at the current noise level
            noise_shape_div = (noise_shape_div_h, noise_shape_div_w, noise_shape[2])

            if USE_NORMAL_NOISE_JUST_FOR_DEMONSTRATION:
                # random normal noise
                gaussian_noise_img_lowres = np.clip(np.random.normal(loc=0, scale=255, size=noise_shape_div).astype(np.float32) / 255, 0.0, 1.0)
            else:
                # style-based noise
                gaussian_noise_img_lowres = make_style_noise(style_img_levels[0], noise_shape_div)

            # high-resolution noise map for the current noise level without gaussian envelope
            gaussian_noise_img_level = cv2.resize(gaussian_noise_img_lowres, dsize=(nw, nh),
                                                  interpolation=cv2.INTER_CUBIC)

            if WITHOUT_GAUSSIAN_MASK_JUST_FOR_DEMONSTRATION:
                # accumulating the multi-level noise map without gaussian envelope
                gaussian_noise_img += gaussian_noise_img_level
            else:
                # accumulating the multi-level noise map with gaussian envelopes
                noise_level_gauss_mask = gaussian_mask(gaussian_noise_img_level.shape, central_amplitude, peripheral_amplitude, dispersion_scale)
                gaussian_noise_img += gaussian_noise_img_level * noise_level_gauss_mask

    if SHOW_TEST_IMGS:
        # showing the obtained noise map
        temp_img = copy.deepcopy(gaussian_noise_img)
        #temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("noise_mask.jpg", temp_img * 255)

        # showing a blurry version of the obtained noise map
        temp_img = cv2.GaussianBlur(temp_img, (107, 107), 0)
        cv2.imwrite("noise_mask_blurry.jpg", temp_img * 255)

    if IGNORE_GRADIENT_MAP_JUST_FOR_DEMONSTRATION:
        # calculating a weight factor for the noise map
        noise_replacement = noise_factor
    else:
        # taking into account the absolute value of the local gradient
        # computing gradients along the X and Y axis, respectively
        sobelX = cv2.Sobel(content_img_levels[0], cv2.CV_64F, 1, 0, ksize=5)
        sobelY = cv2.Sobel(content_img_levels[0], ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
        sobelX = np.absolute(sobelX)
        sobelY = np.absolute(sobelY)

        sobelCombined = np.sqrt(sobelX*sobelX + sobelY*sobelY)
        sobelCombined = np.clip(sobelCombined, 0.0, 100)

        # getting a blurred gradient mask
        sobelCombined = cv2.GaussianBlur(sobelCombined, ksize=(101, 101), sigmaX=0.2)
        # calculating a weight factor for the noise map
        a = 5.0
        noise_replacement = a * noise_factor / (a + sobelCombined)

        if SHOW_TEST_IMGS:
            # showing the gradient mask
            cv2.imwrite("test_noise_rep_blurry.jpg", noise_replacement * 255)

    # choosing an initial image for optimization
    if init_method == 'random':
        init_img_next = gaussian_noise_img * 0.5
        init_img_name = 'random'
    elif init_method == 'content+noise':
        init_img_next = await resize(content_n_style.content[1], level=level)
        init_img_name = content_n_style.content[0]
        # the best choice so far
        init_img_next = ((1.0 - noise_replacement) * init_img_next + noise_replacement * gaussian_noise_img).astype(np.float32)
    else:
        # init image has same dimension as content image - this is a hard constraint
        style_img_resized = await resize(content_n_style.style[1], level=level)
        init_img_next = style_img_resized
        init_img_name = content_n_style.style[0]

    nst = NeuralStyleTransfer(device, model_name, style_img_levels, optimizer_name)
    # the main processing loop
    print("entering processing loop")
    lr_start = 10.0
    async for img, cur_iter in nst.process(content_img_levels, init_img_next, lr_start, iters_num, content_weight, style_weight, tv_weight, init_img_name):
        # calculating current progress
        percent = cur_iter / iters_num * 100.0
        cur_iter += 1
        yield percent, img


def prepare_img(img, device):
    """ A function for preparing and normalization of input images using ImageNet's statistics """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    return transform(img.copy()).to(device).unsqueeze(0)


def unprepare_img(img: Tensor):
    """ A function for reversing of the prepare_img() effect """
    dump_img = img.permute([0, 2, 3, 1]).squeeze(0).to("cpu").detach().numpy()
    dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))

    dump_img = dump_img.astype(np.float32) / 255

    return dump_img


def gaussian_mask(shape, central_amplitude, peripheral_amplitude, dispersion_scale=0.5):
    """ A function for obtaining a gaussian envelope for the noise map """
    # Extracting the height and width of an image
    rows, cols = shape[:2]
    nw = shape[1]
    nh = shape[0]

    # generating vignette mask using Gaussian resultant_kernels
    X_resultant_kernel = cv2.getGaussianKernel(cols, nw * dispersion_scale)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, nh * dispersion_scale)

    # generating resultant_kernel matrix
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T

    gauss_norm = resultant_kernel / resultant_kernel[rows // 2, cols // 2]

    # The gaussian mask function shall rise from central_amplitude in the center to (almost) 1.0 on the edges
    mask = peripheral_amplitude + gauss_norm * (central_amplitude - peripheral_amplitude)

    # applying the mask to each channel of the input image
    expanded = np.expand_dims(mask, 2)
    res = np.repeat(expanded, 3, axis=2)
    return res



def make_style_noise(style_img_np, targ_shape):
    """ A function for making a noise map by randomly permuting the pixels of the input style image """
    nw = targ_shape[1]
    nh = targ_shape[0]
    inp_img_copy = copy.deepcopy(style_img_np)
    style_img_np_resized = cv2.resize(inp_img_copy, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)

    style_vect = style_img_np_resized.reshape(nh * nw, -1)
    style_noise_vect = np.random.permutation(style_vect)

    res = style_noise_vect.reshape(targ_shape)

    # res[:, :, 0] = (res[:,:,0] + res[:,:,1] + res[:,:,2]) / 3
    # res[:, :, 1] = res[:,:,0]
    # res[:, :, 2] = res[:,:,0]

    #cv2.imwrite("test.jpg", res * 255)
    return res

