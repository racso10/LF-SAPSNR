import torch.nn.functional as F
import numpy as np
import torch
import cv2
import skimage.color as color
import math
import re
from einops import rearrange
import lpips
import h5py


def overlap_crop_forward(x, position, margin, offset_k, model, max_length=512, shave=10, mod=1):
    """
    chop for less memory consumption during test
    """
    n_GPUs = 1
    if len(x.shape) == 6:
        b, u, v, h, w, c = x.size()
    else:
        b, u, v, h, w = x.size()
    h_half, w_half = h // 2, w // 2

    h_size, w_size = int(math.ceil((h_half + shave) / mod) * mod), int(math.ceil((w_half + shave) / mod) * mod)
    lr_list = [
        x[:, :, :, 0:h_size, 0:w_size],
        x[:, :, :, 0:h_size, (w - w_size):w],
        x[:, :, :, (h - h_size):h, 0:w_size],
        x[:, :, :, (h - h_size):h, (w - w_size):w]]

    sr_list = []
    for i in range(0, 4, n_GPUs):
        lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
        if lr_batch.shape[3] > max_length or lr_batch.shape[4] > max_length:
            sr_batch_temp = overlap_crop_forward(lr_batch, position, margin, offset_k, model, max_length, shave=shave)
        else:
            sr_batch_temp = model(lr_batch, position, None, margin)

        if isinstance(sr_batch_temp, list):
            sr_batch = sr_batch_temp[-1]
        else:
            sr_batch = sr_batch_temp

        sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))

    n = sr_list[0].shape[1]
    output = torch.zeros(b, n, h, w)
    output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


class LPIPS():
    def __init__(self):
        self.loss_fn = lpips.LPIPS(net='alex').cuda()  # best forward scores
        # self.loss_fn = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

    def compute_lpips(self, img0, img1):
        img0 = torch.from_numpy(img0).cuda()
        img1 = torch.from_numpy(img1).cuda()
        img0 = (img0 - 0.5) * 2
        img1 = (img1 - 0.5) * 2
        img0 = rearrange(img0, 'h w (b c) -> b c h w', b=1)
        img1 = rearrange(img1, 'h w (b c) -> b c h w', b=1)
        result = self.loss_fn(img0, img1)
        result = result.squeeze().item()
        return result


def get_position(view_n_new):
    position = torch.zeros(2, view_n_new * view_n_new)
    for i in range(view_n_new):
        for j in range(view_n_new):
            position[0, i * view_n_new + j] = i
            position[1, i * view_n_new + j] = j
    return position


def get_position_distribute_grid(position, margin):
    margin = np.array(margin)
    margin_central = (margin[1:] + margin[:-1]) / 2
    position_dict = {}
    for i in range(position.shape[1]):
        central_u = np.argmin(np.fabs(margin_central - position[0, i].item()))
        central_v = np.argmin(np.fabs(margin_central - position[1, i].item()))
        position[0, i] -= margin[central_u]
        position[1, i] -= margin[central_v]
        if (central_u, central_v) in position_dict:
            position_dict[(central_u, central_v)].append(position[:, i].unsqueeze(1))
        else:
            position_dict[(central_u, central_v)] = [position[:, i].unsqueeze(1)]

    return position_dict


def image_prepare_HCI(image_path, view_n_ori, view_n_out, is_multi_channel=False):
    LF_img = []
    for i in range(view_n_ori * view_n_ori):
        image_bgr = cv2.imread(image_path + f'input_Cam{str(i).zfill(3)}.png', cv2.IMREAD_UNCHANGED)
        image_bgr = image_bgr[:, :, 0:3]
        image_rgb = image_bgr[:, :, ::-1]

        if not is_multi_channel:
            LF_img.append(color.rgb2ycbcr(image_rgb))
        else:
            LF_img.append(image_rgb)
    LF_img = np.array(LF_img, dtype=np.float32)
    LF_img = rearrange(LF_img, '(u v) h w c -> u v h w c', u=view_n_ori, v=view_n_ori)
    view_n_start = (view_n_ori + 1 - view_n_out) // 2
    LF_img = LF_img[view_n_start:view_n_start + view_n_out, view_n_start:view_n_start + view_n_out]

    return LF_img


def image_prepare_HCI_old(image_path, view_n_ori, view_n_out, is_multi_channel=False):
    with h5py.File(image_path + '.h5', 'r') as hf:
        LF = np.array(hf.get('LF'))

    LF = np.flip(LF, axis=1)

    view_n_start = (view_n_ori + 1 - view_n_out) // 2
    LF = LF[view_n_start:view_n_start + view_n_out, view_n_start:view_n_start + view_n_out]

    LF_img = []
    if not is_multi_channel:
        for i in range(view_n_out):
            for j in range(view_n_out):
                LF_img.append(color.rgb2ycbcr(LF[i, j]))

    LF_img = np.array(LF_img, dtype=np.float32)
    LF_img = rearrange(LF_img, '(u v) h w c -> u v h w c', u=view_n_out, v=view_n_out)

    return LF_img


def image_prepare_DLFD(image_path, view_n_ori, view_n_out, is_multi_channel=False):
    LF_img = []
    view_n_start = (view_n_ori + 1 - view_n_out) // 2
    for i in range(view_n_start, view_n_start + view_n_out):
        for j in range(view_n_start, view_n_start + view_n_out):
            image_bgr = cv2.imread(image_path + f'/lf_{i + 1}_{j + 1}.png', cv2.IMREAD_UNCHANGED)
            image_bgr = image_bgr[:, :, 0:3]
            image_rgb = image_bgr[:, :, ::-1]

            if not is_multi_channel:
                LF_img.append(color.rgb2ycbcr(image_rgb))
            else:
                LF_img.append(image_rgb)
    LF_img = np.array(LF_img, dtype=np.float32)
    LF_img = rearrange(LF_img, '(u v) h w c -> u v h w c', u=view_n_out, v=view_n_out)

    return LF_img


def image_prepare_Lytro(image_path, view_n_ori, view_n_out, is_multi_channel=False):
    image_bgr = cv2.imread(image_path + '.png', cv2.IMREAD_UNCHANGED)
    image_bgr = image_bgr[:, :, 0:3]
    image_rgb = image_bgr[:, :, ::-1]

    image_rgb = rearrange(image_rgb, '(h u) (w v) c -> u v h w c', u=view_n_ori, v=view_n_ori)
    view_n_start = (view_n_ori + 1 - view_n_out) // 2
    image_rgb = image_rgb[view_n_start:view_n_start + view_n_out, view_n_start:view_n_start + view_n_out]

    LF_img = []
    for i in range(view_n_out):
        for j in range(view_n_out):
            if not is_multi_channel:
                LF_img.append(color.rgb2ycbcr(image_rgb[i, j]))
            else:
                LF_img.append(image_rgb[i, j])

    LF_img = np.array(LF_img, dtype=np.float32)
    LF_img = rearrange(LF_img, '(u v) h w c -> u v h w c', u=view_n_out, v=view_n_out)

    return LF_img


def image_prepare_ycbcr(gt_image_ycbcr, margin, view_n_in):
    image_h = gt_image_ycbcr.shape[2]
    image_w = gt_image_ycbcr.shape[3]

    lr_y = np.zeros((view_n_in, view_n_in, image_h, image_w), dtype=np.float32)
    lr_cbcr = np.zeros((view_n_in, view_n_in, image_h, image_w, 2), dtype=np.float32)

    for i in range(view_n_in):
        for j in range(view_n_in):
            lr_y[i, j, :, :] = gt_image_ycbcr[margin[i], margin[j], :, :, 0]
            lr_cbcr[i, j, :, :, :] = gt_image_ycbcr[margin[i], margin[j], :, :, 1:3]

    return lr_y / 255.0, gt_image_ycbcr[:, :, :, :, 0] / 255.0, lr_cbcr


def image_prepare_rgb(gt_image_rgb, margin, view_n_in):
    image_h = gt_image_rgb.shape[2]
    image_w = gt_image_rgb.shape[3]

    lr_rgb = np.zeros((view_n_in, view_n_in, image_h, image_w, 3), dtype=np.float32)

    for i in range(0, view_n_in, 1):
        for j in range(0, view_n_in, 1):
            lr_rgb[i, j, :, :] = gt_image_rgb[margin[i], margin[j]]

    return lr_rgb / 255.0, gt_image_rgb / 255.0


def image_input_asr_fast(dataset, image_path, view_n_ori, view_n_out, n_shear=0, is_multi_channel=False):
    if dataset == 'Syn':
        gt_image = image_prepare_HCI(image_path, view_n_ori, view_n_out, is_multi_channel=is_multi_channel)
    elif dataset == 'Lytro':
        gt_image = image_prepare_Lytro(image_path, view_n_ori, view_n_out, is_multi_channel=is_multi_channel)
    else:
        return None
    if not is_multi_channel:
        gt_image = gt_image[..., 0]

    gt_image = gt_image / 255.0

    if dataset == 'Syn':
        gt_image_input_list = []
        disparity_list = [i for i in range(-n_shear, n_shear + 1)]
        gt_image = torch.from_numpy(gt_image.copy()).cuda()
        if is_multi_channel:
            img_shear = shear_all_multi_channel(img=gt_image, disparity_list=disparity_list)
        else:
            img_shear = shear_all(img=gt_image, disparity_list=disparity_list)
        img_shear = img_shear.cpu().numpy()
        for i, disparity in enumerate(disparity_list):
            img_shear_item = img_shear[i]
            cut = abs(view_n_out // 2 * disparity)
            if cut != 0:
                img_shear_item = img_shear_item[:, :, cut:-cut, cut:-cut]
            gt_image_input_list.append(img_shear_item)
        return gt_image_input_list
    elif dataset == 'Lytro':
        return gt_image


def shear_all(img, disparity_list):
    U, V, X, Y = list(img.shape)
    view_n = U
    D = len(disparity_list)
    img = img.reshape(U * V, 1, X, Y)  # UV ,1, X, Y
    view_central = view_n // 2
    img_all = []
    target_position = np.array([view_central, view_central])

    for disparity in disparity_list:
        theta = []
        for i in range(view_n):
            for j in range(view_n):
                ref_position = np.array([i, j])
                d = (target_position - ref_position) * disparity * 2
                theta_t = torch.FloatTensor([[1, 0, d[1] / img.shape[3]], [0, 1, d[0] / img.shape[2]]])
                theta.append(theta_t.unsqueeze(0))
        theta = torch.cat(theta, 0).cuda()
        grid = F.affine_grid(theta, img.size(), align_corners=False)
        img_tmp = F.grid_sample(img, grid, align_corners=False)
        img_tmp = img_tmp.unsqueeze(0)
        img_all.append(img_tmp)
    img_all = torch.cat(img_all, 0)
    img_all = img_all.reshape(D, U, V, X, Y)
    return img_all


def shear_all_multi_channel(img, disparity_list):
    U, V, X, Y, C = list(img.shape)
    view_n = U
    img = rearrange(img, 'u v h w c -> (u v) c h w')
    view_central = view_n // 2
    img_all = []
    target_position = np.array([view_central, view_central])

    for disparity in disparity_list:
        theta = []
        for i in range(view_n):
            for j in range(view_n):
                ref_position = np.array([i, j])
                d = (target_position - ref_position) * disparity * 2
                theta_t = torch.FloatTensor([[1, 0, d[1] / img.shape[3]], [0, 1, d[0] / img.shape[2]]])
                theta.append(theta_t.unsqueeze(0))
        theta = torch.cat(theta, 0).cuda()
        grid = F.affine_grid(theta, img.size(), align_corners=False)
        img_tmp = F.grid_sample(img, grid, align_corners=False)
        img_tmp = img_tmp.unsqueeze(0)
        img_all.append(img_tmp)
    img_all = torch.cat(img_all, 0)
    img_all = rearrange(img_all, 'd (u v) c h w -> d u v h w c', u=U)
    return img_all


def get_parameter_number(net):
    print(net)
    parameter_list = [p.numel() for p in net.parameters()]
    print(parameter_list)
    total_num = sum(parameter_list)
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})
