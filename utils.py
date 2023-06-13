import numpy as np
import pickle
import torch
import os
import math
from torch.nn import DataParallel
import torchvision.transforms as transforms
from torch2trt import torch2trt
import cv2

def load_model(model_path, weight_path, input_shape=[3, 224, 224], model_trt=False, device='cpu', batch_size_trt=128):
    with open(model_path, 'rb') as m_file:
        try:
            model = pickle.load(m_file)
        except Exception as e:
            model = torch.load(m_file)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()
    device_ids = np.arange(torch.cuda.device_count()).tolist() if device.type != 'cpu' else []
    if model_trt:
        dot_index = weight_path.rindex(".")
        model_trt_name = weight_path[:dot_index] + "_trt" + weight_path[dot_index:]
        x = torch.ones([1] + input_shape).to(device)
        if os.path.exists(model_trt_name):
            from torch2trt import TRTModule
            model = TRTModule()
            model.load_state_dict(torch.load(model_trt_name))
            print("Loaded the trt model.")
        else:

            print("Converting the model to a trt model...")
            model = torch2trt(model, [x], max_batch_size=math.ceil(batch_size_trt / len(device_ids)))
            torch.save(model.state_dict(), model_trt_name)
            print("The trt model saved at {}".format(model_trt_name))

    # if len(device_ids) > 1:
    #     model = DataParallel(model, device_ids=device_ids)
    return model


def compute_norm_4d(img1, img2, norm=2):
    """
    input: in torch format
    """
    return torch.linalg.vector_norm((img1 - img2), norm, dim=(1, 2, 3))

def compute_norm(img1, img2, norm=2):
    """
    input: in torch format
    """
    if img1.dim() < 4:
        return torch.norm((img1 - img2), norm)
    else:
        norms = torch.linalg.vector_norm((img1 - img2), norm, dim=(1, 2, 3))
        return norms[0] if len(norms) == 1 else norms

def clip_channel_wise_4d(input, mins, maxs):
    """

    :param input:
    :param mins:
    :param maxs:
    :return:
    """
    input[:, 0, :, :] = torch.clamp(input[:, 0, :, :], mins[0], maxs[0])
    input[:, 1, :, :] = torch.clamp(input[:, 1, :, :], mins[1], maxs[1])
    input[:, 2, :, :] = torch.clamp(input[:, 2, :, :], mins[2], maxs[2])
    # return input.squeeze(0)
    return input


def save_img(img, filename, cmap=None) -> object:

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    # import matplotlib
    #
    # matplotlib.use('Agg')
    # from matplotlib import pyplot as plt
    # plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
    # plt.axis('off')
    # # plt.show()
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    # plt.close()
    cv2.imwrite(filename, img*255)


def topk(x, k):
    # idx = np.argpartition(x, -k)[-k:]  # indices not sorted
    # return idx[np.argsort(x[idx])][::-1]  # indices sorted by value from largest to smallest
    return torch.topk(x, k).indices


def get_delta_prob_dilutions(patch_prob_vectors, gt_idxes, patch_prob_vector, gt_idx):
    prob_dilution_vector = calc_prob_dilution_vec(patch_prob_vectors, gt_idxes)
    prob_dilution = calc_prob_dilution(patch_prob_vector, gt_idx)

    delta_prob_dilutions = prob_dilution_vector - prob_dilution  # low=(self.min_pd - self.max_pd), high=(self.max_pd - self.min_pd)
    return delta_prob_dilutions


def calc_prob_dilution(prob_vector, gt_idx, use_log_prob=True, return_prob=False):
    _prob_vector = torch.clone(prob_vector)
    if return_prob:
        return prob_vector[gt_idx]
    if use_log_prob:
        # prob_dilution = - 1/log(1/p_g) + 1/log(1/p_k1) + 1/log(1/p_k2) + ... + 1/log(1/p_kn)
        assert torch.round(torch.sum(_prob_vector), decimals=2) <= 1.
        # _prob_vector[_prob_vector == 0] += torch.finfo(torch.float32).eps  # add a very small value to zero elements
        # _prob_vector = torch.stack(
        #     [transform_fun(_prob_vector[i], use_ln=False, k=200) for i in range(len(_prob_vector))])
        # _prob_vector[gt_idx] = -_prob_vector[gt_idx]
        # prob_dilution = torch.sum(_prob_vector)
        # return prob_dilution
        return 1/(torch.log10(prob_vector[gt_idx]-1.0133e-06)if prob_vector[gt_idx]==1 else torch.log10(prob_vector[gt_idx]))

    else:
        _prob_vector = torch.cat([_prob_vector[:gt_idx], _prob_vector[gt_idx + 1:]])
        _prob_vector = torch.sort(_prob_vector)  # get the second and third top predictions
        prob_dilution = ((prob_vector[gt_idx] + _prob_vector[-1] + _prob_vector[-2]) /
                         prob_vector[gt_idx])
        return prob_dilution


def calc_prob_dilution_vec(prob_vectors, gt_idx):
    """

    :param prob_vectors:
    :param gt_idx:
    :return:
    """
    # if len(gt_idx) == 1:
    #     gt_idx = torch.stack([gt_idx.squeeze()] * len(prob_vectors))
    # print(len(prob_vectors), len(gt_idx))
    prob_dilution_vector = torch.stack(
        [calc_prob_dilution(prob_vectors[i], gt_idx[i]) for i in range(len(prob_vectors))])
    return prob_dilution_vector

def transform_fun(prob, use_ln=True, k=50):
    assert prob <= 1.0
    if use_ln:
        # constant = 50
        log_ = torch.log
        const = 9.4912  # 1 / (torch.log(torch.tensor(1 / 0.9)))
    else:
        # constant = 200
        log_ = torch.log10
        const = 21.8543  # 1 / (log_(torch.tensor(1 / 0.9)))
    output = 1 / (log_(1 / prob)) if prob <= 0.9 else const + (prob - 0.9) * k

    return output


def get_top_prob_vectors(patch_prob_vectors, patch_prob_vector, gt_idx, n=10):
    _patch_prob_vector = patch_prob_vector.squeeze()
    top_p_idx = torch.argsort(_patch_prob_vector, descending=True)[:n]
    _patch_prob_vector = _patch_prob_vector[top_p_idx]

    top_ps_idx = torch.argsort(patch_prob_vectors, dim=1, descending=True)[:, :n]
    gt_idxes = torch.where(top_ps_idx == gt_idx)[1]
    _patch_prob_vectors = torch.stack([patch_prob_vectors[i, r] for i, r in enumerate(top_ps_idx)])
    new_gt_idxes = torch.where(top_p_idx == gt_idx)[0]
    if new_gt_idxes.nelement() == 0:
        raise Exception("True label is not among the top {}. You should increase the top n!".format(n))
    return _patch_prob_vectors, _patch_prob_vector, new_gt_idxes[0], gt_idxes


def to_original_format(img, tensor_format=False):
    """
    converts data from pytorch acceptable format(for images) (channel, height, width) -> (height, width, channel) and converts to numpy nd array
    input: tensor
    output: numpy nd array
    """
    if img.dim() == 4:
        img = img.permute(0, 2, 3, 1)
        img = img.squeeze(0)
        if not tensor_format:
            img = img.cpu().numpy() if img.is_cuda else img.numpy()
    else:
        img = img.permute(1, 2, 0)
        if not tensor_format:
            img = img.cpu().numpy() if img.is_cuda else img.numpy()
    return img


def un_standardize(tensor, mean, std, clip01=False):
    """
    un_standardize data with mean and standard deviation and converts to original distribution.
    input: input tensor, mean, std
    output: un_standardize tensor
    """
    inv_normalize = transforms.Normalize(
        mean=list(-1 * np.array(mean) / np.array(std)),
        std=list(1 / np.array(std))
    )
    inv_tensor = inv_normalize(tensor)

    if clip01:
        inv_tensor = torch.clip(inv_tensor, 0, 1)
    return inv_tensor


def remove_elements(tensor, elements):
    for el in elements:
        tensor = tensor[tensor != el]
    return tensor


class Logger:
    def __init__(self, path):
        self.path = path
        if path != '':
            folder = '/'.join(path.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)

    def print(self, message, console=True):
        if console:
            print(message)
        if self.path != '':
            with open(self.path, 'a') as f:
                f.write(message + '\n')
                f.flush()


def save_step(dc, f_name):
    f_name = f_name + ".pkl" if f_name.split(".")[-1] != 'pkl' else f_name
    with open(f_name, 'wb') as f:
        pickle.dump(dc, f, pickle.HIGHEST_PROTOCOL)
