import copy
import os

import cv2

MYPATH = os.path.dirname(os.path.abspath(__file__))
print(f"Script's directory: {MYPATH}")
os.environ["TORCH_HOME"] = MYPATH

import utils

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import argparse
import asyncio


class ContentStylePair:
    def __init__(self, content, style):
        self.content = content
        self.style = style


class RepresentationBuilder:
    def __init__(self, image, neural_net):
        self.__image = image
        self.__neural_net = neural_net

        self.__features = neural_net(image)

    def build_content(self, feature_map_indices: int | list[int]):
        list_taken = isinstance(feature_map_indices, list)
        indices = feature_map_indices if list_taken else [ feature_map_indices ]

        rep = [ x.squeeze(axis=0) for index, x in enumerate(self.__features) if index in indices ]

        return rep if list_taken else rep[0]

    def build_style(self, feature_map_indices: int | list[int]):
        list_taken = isinstance(feature_map_indices, list)
        indices = feature_map_indices if list_taken else [ feature_map_indices ]

        rep = [ utils.gram_matrix(x) for index, x in enumerate(self.__features) if index in indices ]

        return rep if list_taken else rep[0]


class LossBuilder:
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
        content_loss = torch.nn.MSELoss(reduction='mean')(self.__target_content_representation, current_content_representation)

        current_style_representation = current_rep_builder.build_style(self.__style_feature_maps_indices)

        # Calculating style representation for hires
        style_loss = 0.0
        for gram_gt, gram_hat in zip(self.__target_style_representation, current_style_representation):
            style_loss += torch.nn.MSELoss(reduction='mean')(gram_gt[0], gram_hat[0])
        style_loss /= len(self.__target_style_representation)

        tv_loss = utils.total_variation(optimizing_img) # + utils.regularization(optimizing_img)

        total_loss = self.__content_weight * content_loss + self.__style_weight * style_loss + self.__tv_weight * tv_loss

        return total_loss, content_loss, style_loss, tv_loss


class NeuralStyleTransfer:
    def __init__(self, device, model_name, style_imgs, optimizer_name):
        self.__device = device
        self.__model_name = model_name
        self.__style_imgs = style_imgs
        self.__optimizer_name = optimizer_name

    async def process(self, content_imgs, init_img, lr_start, iters_num, content_weight, style_weight, tv_weight, init_img_name):
        neural_net, content_feature_maps_index, style_feature_maps_indices = utils.prepare_model(self.__model_name, self.__device)
        print(f'Using {self.__model_name} in the optimization procedure.')

        init_img = utils.prepare_img(init_img, self.__device)
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
            content_img = utils.prepare_img(content_img, self.__device)
            style_img = utils.prepare_img(style_img, self.__device)

            loss_builders.append(LossBuilder(content_feature_maps_index, style_feature_maps_indices, content_img, style_img,
                                            neural_net, content_weight, style_weight, tv_weight))

        step = 0
        torch.autograd.set_detect_anomaly(True)

        def optimizer_step_callback():
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
                    optimizing_img_levels = [optimizing_img]
                else:
                    sw = optimizing_img_levels[i - 1].shape[3]
                    sh = optimizing_img_levels[i - 1].shape[2]
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

        while step < iters_num:
            await asyncio.get_running_loop().run_in_executor(None, optimizer.step, optimizer_step_callback)
            res_img = copy.deepcopy(optimizing_img)
            yield utils.unprepare_img(res_img), step


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


async def neural_style_transfer(content_n_style: ContentStylePair,
                                content_weight, style_weight, tv_weight,
                                optimizer, model, init_method,
                                iters_num, levels_num):
    print("entering neural_style_transfer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = model
    optimizer_name = optimizer

    level = 0

    content_img_level0 = await resize(content_n_style.content[1], level=level) #utils.load_image(content_img_path, target_shape=height, blur=False)
    style_img_level0 = await resize(content_n_style.style[1], level=level) #utils.load_image(style_img_path, target_shape=height, blur=False)

    # Starting the processing
    content_img_levels = [ content_img_level0 ]
    style_img_levels = [ style_img_level0 ]
    print("entering levels")

    for level in range(1, levels_num):

        content_img_next = await resize(content_n_style.content[1], level=level) #utils.load_image(content_img_path, target_shape=height_next, blur=False)
        style_img_next = await resize(content_n_style.style[1], level=level) #utils.load_image(style_img_path, target_shape=height_next, blur=False)

        content_img_levels.insert(0, content_img_next)
        style_img_levels.insert(0, style_img_next)

    gaussian_noise_img = np.clip(np.random.normal(loc=0, scale=255, size=content_img_levels[0].shape).astype(np.float32) / 255, 0.0, 1.0)

    if init_method == 'random':
        init_img_next = gaussian_noise_img * 0.5
        init_img_name = 'random'
    elif init_method == 'content':
        init_img_next = await resize(content_n_style.content[1], level=level) #utils.load_image(content_img_path, target_shape=height_next, blur=False)
        init_img_name = content_n_style.content[0]
        #init_img_next = 0.5 * (init_img_next + gaussian_noise_img)
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style_img_resized = await resize(content_n_style.style[1], level=level) #utils.load_image(style_img_path, target_shape=np.asarray(content_img_levels[0].shape[2:]), blur=False)
        init_img_next = style_img_resized
        init_img_name = content_n_style.style[0]

    nst = NeuralStyleTransfer(device, model_name, style_img_levels, optimizer_name)
    print("entering processing loop")
    async for img, cur_iter in nst.process(content_img_levels, init_img_next, 10.0, iters_num, content_weight, style_weight, tv_weight, init_img_name):
        print("in the processing loop")
        percent = cur_iter / iters_num * 100.0
        cur_iter += 1
        yield percent, img


def load_image(img_path):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv2.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="content image name", default='figures.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='vg_starry_night.jpg')

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e1)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=1e5)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=0e3)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--iters_num", type=int, help="number of iterations to perform", default=200)
    parser.add_argument("--levels_num", type=int, help="number of pyramid levels", default=3)

    args = parser.parse_args()

    content_img_name = args.content_img_name
    style_img_name = args.style_img_name
    content_weight = args.content_weight
    style_weight = args.style_weight
    tv_weight = args.tv_weight
    optimizer = args.optimizer
    model = args.model
    init_method = args.init_method
    iters_num = args.iters_num
    levels_num = args.levels_num

    # original NST (Neural Style Transfer) algorithm (Gatys et al.)
    content_img_path = os.path.join(content_images_dir, content_img_name)
    style_img_path = os.path.join(style_images_dir, style_img_name)

    content_n_style = ContentStylePair((content_img_name, load_image(content_img_path)), (style_img_name, load_image(style_img_path)))
    results_path = neural_style_transfer(content_n_style, content_weight, style_weight, tv_weight, optimizer, model, init_method, iters_num, levels_num)