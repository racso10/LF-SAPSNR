import torch
import os
import numpy as np
import cv2
from einops import rearrange
import torch.nn.functional as F

from model import LF_SAPSNR
import utils


def test_grid(img_path, image_list, model, view_n_new, view_n_old, view_n_oir, disparity_range):
    stride = (view_n_new - 1) // (view_n_old - 1)
    margin = [i * stride for i in range(view_n_old)]

    for index, image_name in enumerate(image_list):
        gt_image_rgb = utils.image_prepare_HCI(img_path + image_name, view_n_oir,
                                                   view_n_new, is_multi_channel=True)
        lr_rgb, gt_hr_rgb = utils.image_prepare_rgb(gt_image_rgb, margin, view_n_old)

        position_tmp_list = []
        for i in range(view_n_new):
            for j in range(view_n_new):
                position_tmp_list.append([i, j])

        with torch.no_grad():
            lr_rgb = lr_rgb[np.newaxis, :, :, :, :]
            lr_rgb = torch.from_numpy(lr_rgb.copy())
            lr_rgb = lr_rgb.cuda()

            lr_rgb_tmp = model.ori_cnn(lr_rgb)
            lr_rgb_tmp = model.fea_part(lr_rgb_tmp)

            cv2.namedWindow('App Demo')

            param = [model, lr_rgb_tmp, view_n_new, margin]
            cv2.setMouseCallback('App Demo', my_def_view_syn, param)
            cv2.imshow('App Demo', np.zeros((468, 468, 3)))
            cv2.waitKey(0)

            position = torch.zeros(2, view_n_new * view_n_new)
            for i in range(view_n_new):
                for j in range(view_n_new):
                    position[0, i * view_n_new + j] = i
                    position[1, i * view_n_new + j] = j
            hr_rgb = model(lr_rgb, position, None, margin)
            hr_rgb = rearrange(hr_rgb, 'b (u v c) h w -> (b u v) c h w', u=view_n_new, v=view_n_new)
            param = [model, hr_rgb, view_n_new, disparity_range]
            cv2.setMouseCallback('App Demo', my_def_refocus, param)
            cv2.imshow('App Demo', np.zeros((468, 468, 3)))
            cv2.waitKey(0)

    cv2.destroyAllWindows()


def refocus(img, view_n, disparity, DoF, align_corners=False):
    target_position = np.array([view_n // 2, view_n // 2])

    view_list = [i for i in range(view_n)]
    view_list = view_list[view_n // 2 - DoF:view_n // 2 + DoF + 1]

    view_index = []

    theta = []
    for i in view_list:
        for j in view_list:
            ref_position = np.array([i, j])
            d = (target_position - ref_position) * disparity * 2
            theta_t = torch.FloatTensor([[1, 0, d[1] / img.shape[3]], [0, 1, d[0] / img.shape[2]]])
            theta.append(theta_t.unsqueeze(0))
            view_index.append(i * view_n + j)
    theta = torch.cat(theta, 0).cuda()
    grid = F.affine_grid(theta, img[view_index].size(), align_corners=align_corners)
    img_tmp = F.grid_sample(img[view_index], grid, align_corners=align_corners)
    img_tmp = torch.mean(img_tmp, dim=0)
    img_tmp = rearrange(img_tmp, 'c h w -> h w c')

    return img_tmp


def my_def_refocus(event, x, y, flags, param=None):
    model, hr_rgb, view_n_new, disparity_range = param[0], param[1], param[2], param[3]
    disparity = x / hr_rgb.shape[-1] * (disparity_range + 1) * 2 - (disparity_range + 1)
    DoF = int(y / hr_rgb.shape[-2] * view_n_new // 2 + 0.5)
    refcus_results = refocus(hr_rgb, view_n_new, disparity=-disparity, DoF=DoF, align_corners=False)
    refcus_results = refcus_results.cpu().numpy()
    refcus_results = np.clip(refcus_results, 0, 1)

    cv2.imshow('App Demo', refcus_results[22:-22, 22:-22, ::-1])


def my_def_view_syn(event, x, y, flags, param=None):
    model, lr_rgb, view_n_new, margin = param[0], param[1], param[2], param[3]
    x_tmp = model.render_part(lr_rgb.clone(), torch.ones(1) * (view_n_new * (1 - y / lr_rgb.shape[4])),
                              torch.ones(1) * (view_n_new * (1 - x / lr_rgb.shape[5])), margin)
    x_tmp = model.fusion_part(x_tmp)
    hr_rgb = model.encoding_part(x_tmp)

    hr_rgb = rearrange(hr_rgb, 'b c h w -> b h w c')
    hr_rgb = hr_rgb[0].cpu().numpy()
    hr_rgb = np.clip(hr_rgb, 0, 1)

    cv2.imshow('App Demo', hr_rgb[22:-22, 22:-22, ::-1])


def test_main(gpu_no=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)
    torch.backends.cudnn.benchmark = True

    print('done')
    print('=' * 40)
    print('build network...')

    view_n_old = 2
    view_n_new = 9

    model = LF_SAPSNR(is_multi_channel=True)

    utils.get_parameter_number(model)

    model.cuda()
    model.eval()
    print('done')
    print('=' * 40)
    print('load model...')

    dir_model = f'../pretrain_model/GUI.pkl'
    checkpoint = torch.load(dir_model)

    if checkpoint is None:
        print('cannot find model!')
        return

    if 'model' in checkpoint:
        pretrained_model = checkpoint['model']
    else:
        pretrained_model = checkpoint

    model.load_state_dict(pretrained_model)

    print('done')
    print('=' * 40)
    print('predict image...')

    img_path = '/mnt/nfsData10/ChangSong/Dataset/hci_dataset/'
    image_list = []
    for line in open(f'./list/Test_HCI.txt'):
        image_list.append(line.strip())
    view_n_ori = int(image_list[0])
    image_list = image_list[1:]
    image_list.sort()

    # image_list = ['herbs']

    test_grid(img_path, image_list, model, view_n_new, view_n_old, view_n_ori, disparity_range=4)

    print('all done')


if __name__ == '__main__':
    gpu_no = 2
    test_main(gpu_no=gpu_no)
