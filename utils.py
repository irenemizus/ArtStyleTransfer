import cv2 as cv
import math
import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from functools import reduce


from vgg_nets import Vgg16, Vgg19, Vgg16Experimental

IMAGENET_MEAN_255 = [0, 0, 0]
#IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


#
# Image manipulation util functions
#

def load_image(img_path, target_shape=None, blur=False):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the height
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    if blur:
        img = cv.GaussianBlur(img, (3, 3), sigmaX=1.0, sigmaY=1.0)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img




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
    dump_img = img.squeeze(0).to("cpu").detach().numpy()
    dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    # dump_img = np.clip(dump_img, 0, 255).astype(np.float32) / 255
    dump_img = dump_img.astype(np.float32) / 255

    return dump_img




def save_image(img, img_path):
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    cv.imwrite(img_path, img[:, :, ::-1])  # [:, :, ::-1] converts rgb into bgr (opencv contraint...)


def generate_out_img_name(content_img_name, style_img_name, optimizer, init_method, height, model, content_weight, style_weight, tv_weight, img_format):
    prefix = os.path.basename(content_img_name).split('.')[0] + '_' + os.path.basename(style_img_name).split('.')[0]
    # called from the reconstruction script
    #if 'reconstruct_script' in config:
    #    suffix = f'_o_{config["optimizer"]}_h_{str(config["height"])}_m_{config["model"]}{config["img_format"][1]}'
    #else:
    #
    suffix = f'_o_{optimizer}_i_{init_method}_h_{str(height)}_m_{model}_cw_{content_weight}_sw_{style_weight}_tv_{tv_weight}{img_format[1]}'
    return prefix + suffix


def save_and_maybe_display(optimizing_img, name_prefix, dump_path, img_format, saving_freq, img_id, num_of_iterations, content_img_name, style_img_name, optimizer, init_method, height, model, content_weight, style_weight, tv_weight, should_display=False):
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)  # swap channel from 1st to 3rd position: ch, _, _ -> _, _, chr

    # for saving_freq == -1 save only the final result (otherwise save with frequency saving_freq and save the last pic)
    if img_id == num_of_iterations-1 or (saving_freq > 0 and img_id % saving_freq == 0):
        out_img_name = name_prefix + str(img_id).zfill(img_format[0]) + img_format[1] if saving_freq != -1 else generate_out_img_name(content_img_name, style_img_name, optimizer, init_method, height, model, content_weight, style_weight, tv_weight, img_format)
        dump_img = np.copy(out_img)
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')
        cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])

    if should_display:
        plt.imshow(np.uint8(get_uint8_range(out_img)))
        plt.show()


def get_uint8_range(x):
    if isinstance(x, np.ndarray):
        x -= np.min(x)
        x /= np.max(x)
        x *= 255
        return x
    else:
        raise ValueError(f'Expected numpy array got {type(x)}')


#
# End of image manipulation util functions
#


# initially it takes some time for PyTorch to download the models into local cache
def prepare_model(model, device):
    # we are not tuning model weights -> we are only tuning optimizing_img's pixels! (that's why requires_grad=False)
    experimental = True
    if model == 'vgg16':
        if experimental:
            # much more flexible for experimenting with different style representations
            model = Vgg16Experimental(requires_grad=False, show_progress=True)
        else:
            model = Vgg16(requires_grad=False, show_progress=True)
    elif model == 'vgg19':
        model = Vgg19(requires_grad=False, show_progress=True)
    else:
        raise ValueError(f'{model} not supported.')

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names

    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.to(device).eval(), content_fms_index_name[0], style_fms_indices_names[0]


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y):
    return torch.mean(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.mean(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

def regularization(y):
    els = reduce(lambda a, b: a*b, y.shape)
    return torch.sum(torch.pow(y / 128.0, 10)) / math.pow(els, 10)