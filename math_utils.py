import math
import torch
from functools import reduce


from neural_nets import Vgg19

# initially it takes some time for PyTorch to download the models into local cache
def prepare_model(model, device):
    # we are not tuning model weights -> we are only tuning optimizing_img's pixels! (that's why requires_grad=False)
    if model == 'vgg19':
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
    mean_x = torch.mean(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:]))
    mean_y = torch.mean(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
    return (mean_x*mean_x + mean_y*mean_y)  #torch.sqrt


def regularization(y):
    els = reduce(lambda a, b: a*b, y.shape)
    return torch.sum(torch.pow(y / 128.0, 10)) / math.pow(els, 10)