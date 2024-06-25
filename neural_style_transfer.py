import os
MYPATH = os.path.dirname(os.path.abspath(__file__))
print(f"Script's directory: {MYPATH}")
os.environ["TORCH_HOME"] = MYPATH

import utils

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import argparse
#import cv2 as cv


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
    def __init__(self, device, model_name, style_imgs, optimizer_name, img_format):
        self.__device = device
        self.__model_name = model_name
        self.__style_imgs = style_imgs
        self.__optimizer_name = optimizer_name
        #self.__dump_path = dump_path
        self.__img_format = img_format

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
        #pools = []
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
                    #pool_twice = torch.nn.AvgPool2d(kernel_size=2)
                    #pool_twice = torch.nn.functional.avg_pool2d(conv_twice, kernel_size=2)
                    #pools.append(pool_twice)

                    # kernel = torch.tensor(
                    #     [
                    #             [0.25, 0.5, 0.25],
                    #             [0.5,  1.0, 0.5],
                    #             [0.25, 0.5, 0.25]
                    #     ], device=self.__device).reshape(1, 1, 3, 3) / 4.0
                    # kernel = kernel.repeat((3, 1, 1, 1))
                    # conv_twice = torch.nn.functional.conv2d(optimizing_img_levels[i - 1], kernel, groups=3, stride=2, padding=1, bias=torch.tensor([0.0, 0.0, 0.0], device=self.__device))



                    # kernel = torch.tensor(
                    #     [
                    #         [0.25,    0.5, 0.125,   0.5, 0.0625],
                    #         [0.25,   0.25,   0.5,  0.25,   0.25],
                    #         [0.125,   0.5, 0.875,   0.5,  0.125],
                    #         [0.25,   0.25,   0.5,  0.25,   0.25],
                    #         [0.0625,    0.5, 0.125,   0.5, 0.0625]
                    #     ], device=self.__device).reshape(1, 1, 3, 3) / 4.0
                    # kernel = kernel.repeat((3, 1, 1, 1))
                    # conv_twice = torch.nn.functional.conv2d(optimizing_img_levels[i - 1], kernel, groups=3, stride=2, padding=2, bias=torch.tensor([0.0, 0.0, 0.0], device=self.__device))

                    sw = optimizing_img_levels[i - 1].shape[3]
                    sh = optimizing_img_levels[i - 1].shape[2]
                    conv_twice = torch.nn.functional.interpolate(optimizing_img_levels[i - 1], size=(sw // 2, sh // 2), mode='bicubic')

                    optimizing_img_levels.append(conv_twice) #(optimizing_img_levels[i - 1])

                    #debug_print_smaller_images = False
                    #if debug_print_smaller_images:
                    #    utils.save_and_maybe_display(optimizing_img_levels[-1], name_prefix + "small_" + str(i) + "_", self.__dump_path, img_format, saving_freq,
                    #                                 step, iters_num, content_img_name, style_img_name, optimizer, init_method,
                    #                                 height, model, content_weight, style_weight, tv_weight,
                    #                                 should_display=False)

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
                #utils.save_and_maybe_display(optimizing_img, name_prefix, self.__dump_path, img_format, saving_freq, step, iters_num, content_img_name, style_img_name, optimizer, init_method, height, model, content_weight, style_weight, tv_weight, should_display=False) # num_of_iterations[config['optimizer']]

            step += 1
            return total_loss

        while step < iters_num: #num_of_iterations[config['optimizer']]:
            optimizer.step(optimizer_step_callback)
            res_img = optimizing_img.permute([0, 2, 3, 1])
            yield utils.unprepare_img(res_img).copy()


async def neural_style_transfer_fake(content_img_name, style_img_name, height, content_weight, style_weight, tv_weight, optimizer, model, init_method, content_images_dir, style_images_dir, output_img_dir, img_format):
     print("entering neural_style_transfer")
     #IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     content_img_path = os.path.join(content_images_dir, content_img_name)
     print(content_img_path)
     content_img = utils.load_image(content_img_path, target_shape=height, blur=False)
     for i in range(10):
         #print("In NST: " + str(i) + ", " + str(content_img[0][0][0]))
         content_img_prep = utils.prepare_img(content_img, device)
         newImage = content_img_prep.permute([0, 2, 3, 1])
         newImage = utils.unprepare_img(newImage)
         yield (i, newImage)

async def neural_style_transfer(content_img_name, style_img_name, height, content_weight, style_weight, tv_weight, optimizer, model, init_method, content_images_dir, style_images_dir, output_img_dir, img_format):
    print("entering neural_style_transfer")
    content_img_path = os.path.join(content_images_dir, content_img_name)
    style_img_path = os.path.join(style_images_dir, style_img_name)

   # out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
   # dump_path = os.path.join(output_img_dir, out_dir_name)
   # os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    levels_num = 2
    iters_num = 100
    model_name = model
    optimizer_name = optimizer

    # height = int(config['height'])

    #height_level0 = height
    content_img_level0 = utils.load_image(content_img_path, target_shape=height, blur=False)
    style_img_level0 = utils.load_image(style_img_path, target_shape=height, blur=False)

    # Starting the processing
    content_img_levels = [ content_img_level0 ]
    style_img_levels = [ style_img_level0 ]
    #levels = []
    height_next = height
    level = 0
    #levels = [ neural_style_transfer_single(content_img_levels, style_img_levels, init_img_level0, device, model_name, optimizer_name, 10.0, iters_num, dump_path, "l0_", config) ]

    ######################################

    print("entering levels")
    for level in range(1, levels_num):
        #width_next = init_img_level0.shape[1] * 2
        height_next *= 2

        #init_img_next = cv.resize(levels[0], dsize=(height_next, width_next), interpolation=cv.INTER_CUBIC)
        #gaussian_noise_img = np.clip(np.random.normal(loc=0, scale=255, size=init_img_next.shape).astype(np.float32) / 255, 0.0, 1.0)
        #init_img_next = 0.5 * (init_img_next + gaussian_noise_img)

        content_img_next = utils.load_image(content_img_path, target_shape=height_next, blur=False)
        style_img_next = utils.load_image(style_img_path, target_shape=height_next, blur=False)

        content_img_levels.insert(0, content_img_next)
        style_img_levels.insert(0, style_img_next)

    print(content_img_path)
    gaussian_noise_img = np.clip(np.random.normal(loc=0, scale=255, size=content_img_levels[0].shape).astype(np.float32) / 255, 0.0, 1.0)

    if init_method == 'random':
        init_img_next = gaussian_noise_img * 0.5
        init_img_name = 'random'
    elif init_method == 'content':
        init_img_next = utils.load_image(content_img_path, target_shape=height_next, blur=False)
        init_img_name = content_img_name
        #init_img_next = 0.5 * (init_img_next + gaussian_noise_img)
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style_img_resized = utils.load_image(style_img_path, target_shape=np.asarray(content_img_levels[0].shape[2:]), blur=False)
        init_img_next = style_img_resized
        init_img_name = style_img_name

    nst = NeuralStyleTransfer(device, model_name, style_img_levels, optimizer_name, img_format)
    cur_iter = 0
    print("entering processing loop")
    async for img in nst.process(content_img_levels, init_img_next,10.0, iters_num, content_weight, style_weight, tv_weight, init_img_name):
        print("in the processing loop")
        percent = cur_iter / iters_num * 100.0
        cur_iter += 1
        yield percent, img.copy()
    #levels.insert(0, nst.process(content_img_levels, init_img_next,10.0, iters_num, content_weight, style_weight, tv_weight, f"l{level}_"))


if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="content image name", default='figures.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='vg_starry_night.jpg')
    parser.add_argument("--height", type=int, help="height of content and style images", default=512)

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e1)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=1e5)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=0e3)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=-1)
    args = parser.parse_args()

    content_img_name = args.content_img_name
    style_img_name = args.style_img_name
    height = args.height
    content_weight = args.content_weight
    style_weight = args.style_weight
    tv_weight = args.tv_weight
    optimizer = args.optimizer
    model = args.model
    init_method = args.init_method
    saving_freq = args.saving_freq

    # just wrapping settings into a dictionary
    #optimization_config = dict()
    #for arg in vars(args):
    #    optimization_config[arg] = getattr(args, arg)
    #optimization_config['content_images_dir'] = content_images_dir
    #optimization_config['style_images_dir'] = style_images_dir
    #optimization_config['output_img_dir'] = output_img_dir
    #optimization_config['img_format'] = img_format


    # original NST (Neural Style Transfer) algorithm (Gatys et al.)
    #results_path = neural_style_transfer(optimization_config)
    results_path = neural_style_transfer(content_img_name, style_img_name, height, content_weight, style_weight, tv_weight, optimizer, model, init_method, content_images_dir, style_images_dir, output_img_dir, img_format)

    # uncomment this if you want to create a video from images dumped during the optimization procedure
    # create_video_from_intermediate_results(results_path, img_format)
