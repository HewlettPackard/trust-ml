import os
import random
import shutil
import gc
from copy import deepcopy
from datetime import datetime
from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
from skimage.util import random_noise as g_noise
from tqdm.auto import trange

from utils import clip_channel_wise_4d, compute_norm_4d, get_top_prob_vectors, get_delta_prob_dilutions, topk, \
    un_standardize, to_original_format, Logger


class RLAB:
    def __init__(self, classifier, agent_config, device, ds_info):

        self.classifier = classifier
        self.patch_size = agent_config['patch_info']['patch_size']
        self.num_forward_masks = agent_config['patch_info']['num_forward_masks']
        self.num_backward_masks = agent_config['patch_info']['num_backward_masks']
        self.mask_dropout = agent_config['patch_info']['dropout']
        self.batch_size = agent_config['patch_info']['batch_size']
        self.noise_mean = agent_config['noise_info']['mean']
        self.noise_var = agent_config['noise_info']['var']
        self.noise_type = agent_config['noise_info']['type']
        self.noise_kernel_size = agent_config['noise_info']['kernel_size']
        self.noise_method = agent_config['noise_info']['method']
        self.n_forward_steps = agent_config['n_forward_steps']
        self.max_steps = agent_config['max_steps']
        self.log_dir = agent_config['log_img_dir']
        self.verbose = agent_config['verbose']
        self.device = device
        self.dataset_mean = ds_info['mean']
        self.dataset_std = ds_info['std']
        self.set_seed(agent_config['seed'])
        self._check_params()
        self.forward_masks = torch.from_numpy(
            self._generate_forward_masks(self.patch_size, self.num_forward_masks, self.noise_mean, self.noise_var,
                                         dropout=self.mask_dropout, noise_type=self.noise_type)).to(self.device)

        self.backward_masks = torch.from_numpy(
            self._generate_backward_masks(self.patch_size, self.num_backward_masks)).to(self.device)
        self.backward_expanded_mask = torch.ones(3, self.patch_size[0], self.patch_size[1], device=device)

    def generate(self, x, y, idx=None):
        """
        generate an adversarial original_image and return it
        :param y:
        :param x:
        :return:
        """

        self.less_5 = False
        self.less_20 = False
        self.less_30 = False
        self.less_40 = False
        self.l2_at_flip = None
        perturbed_image = deepcopy(x)
        self.gt_idx = deepcopy(y)
        prob_patch, pred, _ = self._predict(x)
        n_classes = prob_patch.shape[1]
        self.prev_gt_prob = prob_patch[0][self.gt_idx][0]
        self.prev_noise_var = self.noise_var
        self.patch_idx_list = torch.tensor([], dtype=torch.int8, device=self.device)
        self.dilution_ratios = torch.tensor([], device=self.device)
        prob_patches = self.compute_prob_patches(perturbed_image, self.noise_type)

        self.prev_l2 = torch.norm(x - perturbed_image)

        # create folder to save step outputs
        gt_idx_cpu = self.gt_idx.item()

        cur_episode_dir = "episode_c{}_{:05d}".format(gt_idx_cpu, idx)
        cur_episode_dir = os.path.join(self.log_dir, cur_episode_dir)
        if os.path.exists(cur_episode_dir):
            shutil.rmtree(cur_episode_dir)
        os.makedirs(cur_episode_dir)
        log_path = os.path.join(cur_episode_dir, "result_c{}_{:05d}.txt".format(gt_idx_cpu, idx))
        log = Logger(log_path)
        log.print("{}, {}, {}".format("delta_l2", "delta_prob", "m_avg"),
                  console=False)
        is_done = False
        for step in trange(self.max_steps, desc="RLAB", disable=False if self.verbose in [1] else True):
            # select the patch places for noises to be added
            self.patch_idxes = self.select_patch(prob_patches, prob_patch, self.gt_idx,
                                                 top_patches=self.n_forward_steps, top_k=min(1000, n_classes))
            self.patch_idx_list = torch.cat((self.patch_idx_list, self.patch_idxes))
            # add noise
            for i in range(self.n_forward_steps):
                self.update_forward_mask(x, perturbed_image, method=self.noise_method)
                perturbed_image = self.add_patch_noise(
                    perturbed_image, self.patch_idxes[i], all_patches=False, noise_type=self.noise_type)

            prob_patches = self.compute_prob_patches(perturbed_image, self.noise_type)
            prob_patch, pred, _ = self._predict(perturbed_image)
            l2 = torch.norm(x - perturbed_image)

            gt_prob = prob_patch[0][self.gt_idx][0]

            self.dilution_ratios = torch.cat(
                (self.dilution_ratios, torch.abs((gt_prob - self.prev_gt_prob) / (l2 - self.prev_l2)).unsqueeze(0)))

            self.prev_gt_prob = gt_prob
            self.prev_l2 = l2
            is_done = self._is_done(pred, self.gt_idx)
            # save the original
            if step == 0:
                file_name = os.path.join(cur_episode_dir, "c{}_x.jpg".format(gt_idx_cpu))
                self.save_img(to_original_format(un_standardize(x, self.dataset_mean, self.dataset_std, clip01=True)),
                              file_name)
            if is_done and self.l2_at_flip is None:
                self.l2_at_flip = l2
            if self.l2_at_flip is not None and is_done:
                if l2 > self.l2_at_flip * 16:
                    file_name = os.path.join(cur_episode_dir, "c{}_x_16_adv.jpg".format(gt_idx_cpu))
                    self.save_img(to_original_format(
                        un_standardize(perturbed_image, self.dataset_mean, self.dataset_std, clip01=True)), file_name)
                    break
                elif l2 >= self.l2_at_flip * 12:
                    file_name = os.path.join(cur_episode_dir, "c{}_x_12_adv.jpg".format(gt_idx_cpu))
                    self.save_img(to_original_format(
                        un_standardize(perturbed_image, self.dataset_mean, self.dataset_std, clip01=True)), file_name)
                elif l2 >= self.l2_at_flip * 10:
                    file_name = os.path.join(cur_episode_dir, "c{}_x_10_adv.jpg".format(gt_idx_cpu))
                    self.save_img(to_original_format(
                        un_standardize(perturbed_image, self.dataset_mean, self.dataset_std, clip01=True)), file_name)
                elif l2 >= self.l2_at_flip * 8:
                    file_name = os.path.join(cur_episode_dir, "c{}_x_8_adv.jpg".format(gt_idx_cpu))
                    self.save_img(to_original_format(
                        un_standardize(perturbed_image, self.dataset_mean, self.dataset_std, clip01=True)), file_name)
                elif l2 >= self.l2_at_flip * 6:
                    file_name = os.path.join(cur_episode_dir, "c{}_x_6_adv.jpg".format(gt_idx_cpu))
                    self.save_img(to_original_format(
                        un_standardize(perturbed_image, self.dataset_mean, self.dataset_std, clip01=True)), file_name)
                elif l2 >= self.l2_at_flip * 4:
                    file_name = os.path.join(cur_episode_dir, "c{}_x_4_adv.jpg".format(gt_idx_cpu))
                    self.save_img(to_original_format(
                        un_standardize(perturbed_image, self.dataset_mean, self.dataset_std, clip01=True)), file_name)
                elif l2 >= self.l2_at_flip * 2:
                    file_name = os.path.join(cur_episode_dir, "c{}_x_2_adv.jpg".format(gt_idx_cpu))
                    self.save_img(to_original_format(
                        un_standardize(perturbed_image, self.dataset_mean, self.dataset_std, clip01=True)), file_name)
                elif l2 == self.l2_at_flip:
                    file_name = os.path.join(cur_episode_dir, "c{}_x_adv.jpg".format(gt_idx_cpu))
                    self.save_img(to_original_format(
                        un_standardize(perturbed_image, self.dataset_mean, self.dataset_std, clip01=True)), file_name)

        n_steps = step if self.max_steps == step else step + 1
        timestamp = str(datetime.now())[:-7]
        log.print("{} l2: {:.2f}, n_steps: {}, is_done: {}".format(timestamp, l2, n_steps, is_done.item()),
                  console=False)
        self.wipe_memory()
        return perturbed_image, is_done, n_steps, l2, 0.0

    def misclassify(self, x, y):
        _, pred, _ = self._predict(x)
        return self._is_done(pred, y)

    def wipe_memory(self):
        for item in [self.gt_idx, self.prev_gt_prob, self.patch_idx_list, self.dilution_ratios,
                     self.prev_l2, self.patch_idxes, self.patch_idx_list, self.dilution_ratios, self.prev_gt_prob,
                     self.prev_l2]:
            item = item.to(torch.device('cpu'))
            del item
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _is_done(y1, y2):
        return torch.logical_not(y1 == y2)

    def save_img(self, img, filename, cmap=None):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        np.save(filename.replace('jpg', 'npy'), img)
        import matplotlib

        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _get_moving_averages(self, window_size=7):
        window_averages = torch.tensor([], device=self.device), torch.tensor([], device=self.device), torch.tensor(
            [], device=self.device)
        for i in range(len(self.dilution_ratios) - window_size + 1):
            window = self.dilution_ratios[i: i + window_size]
            window_averages = torch.cat((window_averages, torch.mean(window).unsqueeze(0)))
        return window_averages

    def _get_moving_avg(self, window_size=7):
        return torch.mean(self.dilution_ratios[-window_size:])

    def _add_noise(self, in_mat, noise_type, chosen_mask=None, all_patches=True):
        if noise_type == "GaussianBlur":
            if all_patches:
                in_mat = in_mat.permute(3, 0, 1, 2)
                masks_view_temp = T.GaussianBlur(self.noise_kernel_size)(in_mat)
                in_mat.mul_(torch.zeros_like(in_mat))
                in_mat.add_(masks_view_temp)
            else:
                in_mat_ = T.GaussianBlur(self.noise_kernel_size)(in_mat)
                in_mat.mul_(torch.zeros_like(in_mat))
                in_mat.add_(in_mat_)

        elif noise_type == "GaussianNoise":
            assert chosen_mask is not None, "chosen_mask can't be None!"
            if all_patches:
                in_mat = in_mat.permute(0, 3, 1, 2)
                in_mat.add_(chosen_mask)
            else:
                in_mat.add_(chosen_mask)
        else:  # "DeadPixel":
            assert chosen_mask is not None, "chosen_mask can't be None!"
            if all_patches:
                in_mat = in_mat.permute(0, 3, 1, 2)
                in_mat.mul_(chosen_mask)
            else:
                in_mat.mul_(chosen_mask)

    def add_patch_noise(self, image, patch_idx=None, all_patches=False, noise_type=""):
        """

        :param noise_type:
        :param image:
        :param patch_idx:
        :param all_patches:
        :return:
        """
        torch.cuda.empty_cache()
        if image.dim() < 4:
            image_ = image.unsqueeze(0).clone()
        else:
            image_ = image.clone()
        clip_min, clip_max = [image_[0, i].min() for i in range(3)], [image_[0, i].max() for i in range(3)]
        B, C, H, W = image_.size()
        num_patches = int((H / self.patch_size[0]) * (W / self.patch_size[1]))
        if all_patches:
            patches = torch.repeat_interleave(image_, num_patches, dim=0)
            choice = torch.randint(0, self.forward_masks.size(0), (1,))[0]
            chosen_mask = self.forward_masks[choice]

            chosen_mask = chosen_mask.expand(num_patches, C, self.patch_size[0], self.patch_size[1])
            chosen_mask = chosen_mask.permute(1, 0, 2, 3)
            unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
            patches = unfold(patches)

            patches = patches.view(num_patches, C, *self.patch_size, num_patches)
            patches = patches.permute(0, 4, 1, 2, 3)

            masks_view = torch.diagonal(patches, offset=0, dim1=0, dim2=1)

            self._add_noise(masks_view, noise_type=noise_type, chosen_mask=chosen_mask, all_patches=all_patches)

            patches = patches.view(num_patches, num_patches, C * self.patch_size[0] * self.patch_size[1])
            patches = patches.permute(0, 2, 1)
            output = nn.functional.fold(patches, output_size=(H, W), kernel_size=self.patch_size,
                                        stride=self.patch_size)
            # channel wised clipping
            output = clip_channel_wise_4d(output, clip_min, clip_max)
        else:
            choice = torch.randint(0, self.forward_masks.size(0), (1,))[0]
            chosen_mask = self.forward_masks[choice]

            unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
            patches = unfold(image_)

            patches = patches.view(C, *self.patch_size, num_patches)
            patches = patches.permute(3, 0, 1, 2)
            self._add_noise(patches[patch_idx], noise_type=noise_type, chosen_mask=chosen_mask, all_patches=all_patches)

            patches = patches.view(1, num_patches, C * self.patch_size[0] * self.patch_size[1])
            patches = patches.permute(0, 2, 1)

            output = nn.functional.fold(patches, output_size=(H, W), kernel_size=self.patch_size,
                                        stride=self.patch_size)
            # channel wised clipping
            output = clip_channel_wise_4d(output, clip_min, clip_max)
        return output

    def remove_patch_noise(self, original_image, perturbed_image, predicted_prob, patch_idx_list):
        if original_image.dim() < 4:
            original_image_ = original_image.unsqueeze(0).clone()
        else:
            original_image_ = original_image.clone()
        if perturbed_image.dim() < 4:
            perturbed_image = perturbed_image.unsqueeze(0).clone()
        else:
            perturbed_image = perturbed_image.clone()

        B, C, H, W = original_image_.size()
        clip_min, clip_max = [original_image_[0, i].min() for i in range(3)], [original_image_[0, i].max() for i in
                                                                               range(3)]
        num_patches = int((H / self.patch_size[0]) * (W / self.patch_size[1]))

        original_image_patches = self._unfold_image(original_image_)
        perturbed_image_patches = self._unfold_image(perturbed_image, all_patches=True)
        chosen_mask = self.backward_expanded_mask.expand(num_patches, C, self.patch_size[0], self.patch_size[1])

        # output dimension (total patches, c, patch_size, patch_size)
        transferred_pixels = torch.mul(original_image_patches, chosen_mask)
        transferred_pixels = transferred_pixels.permute(1, 0, 2, 3)
        inverse_mask = 1 - chosen_mask
        inverse_mask = inverse_mask.permute(1, 0, 2, 3)
        masks_view = torch.diagonal(perturbed_image_patches, offset=0, dim1=0, dim2=1)
        masks_view = masks_view.permute(0, 3, 1, 2)
        masks_view.mul_(inverse_mask)
        masks_view.add_(transferred_pixels)
        perturbed_image_patches = perturbed_image_patches.view(num_patches, num_patches,
                                                               C * self.patch_size[0] * self.patch_size[1])

        perturbed_image_patches = perturbed_image_patches.permute(0, 2, 1)
        reconstructed_images = nn.functional.fold(perturbed_image_patches, output_size=(H, W),
                                                  kernel_size=self.patch_size,
                                                  stride=self.patch_size)
        prob_list = self._get_prob(reconstructed_images)
        predicted_probs = torch.max(prob_list, dim=1)
        delta_probs = predicted_probs.values - predicted_prob

        temp_delta_probs = torch.ones_like(delta_probs, device=self.device) * torch.inf
        temp_delta_probs[patch_idx_list] = delta_probs[patch_idx_list]
        chosen_reverse = torch.argmin(temp_delta_probs)
        perturbed_image = clip_channel_wise_4d(reconstructed_images[chosen_reverse].unsqueeze(0), clip_min, clip_max)
        torch.cuda.empty_cache()
        return perturbed_image

    def compute_prob_patches(self, perturbed_image, noise_type):
        perturbed_images = self.add_patch_noise(perturbed_image, all_patches=True, noise_type=noise_type)
        prob_batches = self._get_prob(perturbed_images)
        return prob_batches

    @staticmethod
    def _generate_forward_masks(patch_size, num_masks, noise_mean, noise_var, dropout=0.5, noise_type='GaussianNoise'):
        channel = 3
        mask_list = []

        for i in range(num_masks):
            tot_elements = patch_size[0] * patch_size[1] * channel
            temp = np.ones(tot_elements)
            indices = np.random.choice(np.arange(temp.size), replace=False,
                                       size=int(temp.size * dropout))
            temp[indices] = 0

            if noise_type == 'GaussianNoise':
                noise = np.empty(temp.shape)
                noise = g_noise(noise, mode='gaussian', clip=True, mean=noise_mean, var=noise_var)

                temp = np.multiply(temp, noise)
                temp = np.reshape(temp, (channel, patch_size[0], patch_size[1]))
            else:  # having 0/1 values (all channels are the same)
                temp = np.reshape(temp, (channel, patch_size[0], patch_size[1]))
                temp[1] = temp[2] = temp[0]

            mask_list.append(temp)

        return np.asarray(mask_list)

    @staticmethod
    def select_patch(patch_prob_vectors, patch_prob_vector, gt_idx, top_patches=5, top_k=10):
        # reduce the number of probability vectors for the speed up
        topn_prob_vectors, topn_prob_vector, gt_idx_, gt_idxes = get_top_prob_vectors(patch_prob_vectors,
                                                                                      patch_prob_vector, gt_idx,
                                                                                      n=top_k)
        delta_prob_dilutions = get_delta_prob_dilutions(topn_prob_vectors, gt_idxes, topn_prob_vector, gt_idx_)
        top_patch_idxes = topk(delta_prob_dilutions, top_patches)
        return top_patch_idxes

    def update_adaptive_forward_mask(self, perturbed_image, max_noise=0.5):
        factor = self.noise_var / 5  # 100 max number of steps
        t_noise = self.prev_noise_var
        if len(self.dilution_ratios) < 1:
            return
        m_avg = self._get_moving_avg()
        if self.dilution_ratios[-1] > 0.8 * m_avg:
            t_noise += self.prev_noise_var * factor
            self.prev_noise_var = min(t_noise, max_noise)
            # update forward mask
            self.forward_masks = torch.from_numpy(
                self._generate_forward_masks(self.patch_size, self.num_forward_masks, self.noise_mean,
                                             self.prev_noise_var,
                                             dropout=self.mask_dropout, noise_type=self.noise_type)).to(self.device)

    def update_forward_mask(self, original_image, perturbed_image, method, max_noise=0.5, min_noise=0.005):
        if method.lower() == 'adaptive':
            self.update_adaptive_forward_mask(perturbed_image, max_noise=min_noise)
        elif method.lower() == 'max-ratio':
            self.update_max_forward_mask(original_image, perturbed_image, max_noise=max_noise, min_noise=min_noise)
        else:  # if the method is regular; no change needed.
            return

    def update_max_forward_mask(self, original_image, perturbed_image, max_noise=0.5, min_noise=0.005):
        factor = self.noise_var / 5  # 100 max number of steps
        if len(self.dilution_ratios) < 1:
            return

        noise_values = [max(self.prev_noise_var - self.prev_noise_var * factor, min_noise), self.prev_noise_var,
                        min(self.prev_noise_var + self.prev_noise_var * factor, max_noise)]
        pref_noise = self.prev_noise_var
        max_ratio = -2
        for noise_var in noise_values:
            self.forward_masks = torch.from_numpy(
                self._generate_forward_masks(self.patch_size, self.num_forward_masks, self.noise_mean, noise_var,
                                             dropout=self.mask_dropout, noise_type=self.noise_type)).to(self.device)

            pert_image = self.add_patch_noise(
                perturbed_image, self.patch_idxes[0], all_patches=False, noise_type=self.noise_type)
            prob_patch, _, _ = self._predict(pert_image)
            l2 = torch.norm(original_image - pert_image)
            # record the delta l2 and delta gt prob
            gt_prob = prob_patch[0][self.gt_idx][0]
            ratio = torch.abs((gt_prob - self.prev_gt_prob) / (l2 - self.prev_l2))
            if ratio > max_ratio:
                max_ratio = ratio
                pref_noise = noise_var

        self.prev_noise_var = pref_noise
        self.forward_masks = torch.from_numpy(
            self._generate_forward_masks(self.patch_size, self.num_forward_masks, self.noise_mean, self.prev_noise_var,
                                         dropout=self.mask_dropout, noise_type=self.noise_type)).to(self.device)

    @staticmethod
    def _generate_backward_masks(patch_size, num_masks):
        num_elements = patch_size[0] * patch_size[1]
        if patch_size[0] in [2, 4]:
            mask = np.ones(num_elements)
            masks = np.expand_dims(mask, 0)
            masks = np.repeat(masks, num_masks, axis=0)
        else:
            mask = np.zeros(num_elements)
            masks = np.expand_dims(mask, 0)
            masks = np.repeat(masks, num_masks, axis=0)

            pool = cycle(np.arange(num_masks))
            i = 0
            while i < num_elements:
                arr_id = next(pool)
                masks[arr_id][i] = 1
                i += 1

        masks = np.expand_dims(masks, 1)
        masks = np.repeat(masks, 3, axis=1)
        masks = masks.reshape((num_masks, 3, patch_size[0], patch_size[1]))

        return masks

    def _predict(self, x, k=3):
        if x.dim() < 4:
            x = torch.unsqueeze(x, 0)
        with torch.no_grad():
            outputs = self.classifier(x)

            softmax = torch.nn.Softmax(dim=1)
            outputs = softmax(outputs)
        predicted_class = torch.argmax(outputs, dim=1)
        tks = torch.topk(outputs, k).indices[0]

        return outputs, predicted_class[0], tks

    def _get_prob(self, perturbed_images):
        n_batches = int(np.ceil(perturbed_images.shape[0] / float(self.batch_size)))
        prob_list = torch.tensor([], device=self.device)
        for batch_id in trange(n_batches, desc="Prob calculations", disable=True):
            batch_idx_1, batch_idx_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = perturbed_images[batch_idx_1:batch_idx_2]
            probs, _, _ = self._predict(x_batch)
            prob_list = torch.cat((prob_list, probs))
        return prob_list

    def _get_l2s(self, perturbed_images, original_image):
        n_batches = int(np.ceil(perturbed_images.shape[0] / float(self.batch_size)))
        l2_list = torch.tensor([], device=self.device)
        batch_size = self.batch_size if perturbed_images.size(0) > self.batch_size else perturbed_images.size(0)
        original_image_ = torch.repeat_interleave(
            original_image.unsqueeze(0) if original_image.dim() < 4 else original_image, batch_size, dim=0)
        for batch_id in trange(n_batches, desc="L2 calculations", disable=True):
            batch_idx_1, batch_idx_2 = batch_id * batch_size, (batch_id + 1) * batch_size
            x_batch = perturbed_images[batch_idx_1:batch_idx_2]
            l2_list = torch.cat((l2_list, compute_norm_4d(x_batch, original_image_[:len(x_batch)])), dim=0)

        return l2_list

    @staticmethod
    def _plot_img(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        import matplotlib

        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        _x = to_original_format(un_standardize(x, mean, std, clip01=True))
        plt.imshow(_x)
        plt.show()

    def _unfold_image(self, image, all_patches=False):
        B, C, H, W = image.size()
        num_patches = int((H / self.patch_size[0]) * (W / self.patch_size[1]))

        if all_patches:
            image = torch.repeat_interleave(image, num_patches, dim=0)
            unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
            patches = unfold(image)
            patches = patches.view(num_patches, C, *self.patch_size, num_patches)
            patches = patches.permute(0, 4, 1, 2, 3)

        else:
            unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
            patches = unfold(image)

            patches = patches.view(C, *self.patch_size, num_patches)
            patches = patches.permute(3, 0, 1, 2)

        return patches

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _check_params(self):
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("The max number of steps must be a positive integer.")
