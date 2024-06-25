import cv2 as cv
import math
import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from functools import reduce


from vgg_nets import Vgg16, Vgg19, Vgg16Experimental

#IMAGENET_MEAN_255 = [0, 0, 0]
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
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
    dump_img = img.permute([0, 2, 3, 1]).squeeze(0).to("cpu").detach().numpy()
    dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    # dump_img = np.clip(dump_img, 0, 255).astype(np.float32) / 255
    dump_img = dump_img.astype(np.float32) / 255

    return dump_img


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