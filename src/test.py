import torch
import os
import time
import numpy as np
import pandas as pd
import sys
import skimage.color as color
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from einops import rearrange
from PIL import Image
from argparse import ArgumentParser, ArgumentTypeError

from model import LF_SAPSNR
import utils


class Logger:
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def test_Stanf_grid(img_path, image_list, model, view_n_out, view_n_in, view_n_oir, extra, position, save_path,
                    datasets_name, save_img):
    xls_list = []
    psnr_list = []
    ssim_list = []
    lpips_list = []
    time_list = []

    print(f'===> Task {view_n_in}x{view_n_in} -> {view_n_out}x{view_n_out} Extra {extra}')
    stride = (view_n_out - 1) // (view_n_in - 1)
    margin = [i * stride for i in range(view_n_in)]

    position_dict = utils.get_position_distribute_grid(position, margin)

    loss_fn = utils.LPIPS()

    for index, image_name in enumerate(image_list):
        avg_psnr = []
        avg_ssim = []
        avg_lpips = []
        print('[{}/{}]'.format(index + 1, len(image_list)), image_name)

        gt_image = utils.image_prepare_Lytro(img_path + image_name, view_n_oir, view_n_out, is_multi_channel=True)
        lr, gt_hr = utils.image_prepare_rgb(gt_image, margin, view_n_in)

        pred_hr = torch.zeros(gt_hr.shape).cuda()
        margin_tmp = [0, 4]
        time_ = 0

        for near_grid, position_value in position_dict.items():
            lr_rgb_tmp = lr[near_grid[0]:near_grid[0] + 2, near_grid[1]:near_grid[1] + 2]
            position_tmp = torch.cat(position_value, dim=1)
            hr_rgb_tmp, time_tmp = predict(model, lr_rgb_tmp, position_tmp, margin_tmp, max_length=512)
            time_ += time_tmp
            hr_rgb_tmp = rearrange(hr_rgb_tmp, 'b (n c) h w -> (b n) h w c', c=3)
            for i, position_item in enumerate(position_value):
                pred_hr[int(position_item[0].item()) + margin[near_grid[0]], int(position_item[1].item()) + margin[
                    near_grid[1]]] = hr_rgb_tmp[i]

        pred_hr = pred_hr.cpu().numpy()
        pred_hr = np.clip(pred_hr, 0, 1)
        time_list.append(time_)

        for i in range(view_n_out):
            for j in range(view_n_out):
                if i in margin and j in margin:
                    print('   -   /   -   /   -   ', end='\t\t')
                else:
                    psnr = peak_signal_noise_ratio(pred_hr[i, j, :, :], gt_hr[i, j, :, :], data_range=1.0)
                    ssim = structural_similarity(pred_hr[i, j, :, :], gt_hr[i, j, :, :], win_size=11,
                                                 multichannel=True, gaussian_weights=True)
                    lpips = loss_fn.compute_lpips(pred_hr[i, j, :, :], gt_hr[i, j, :, :])
                    avg_psnr.append(psnr)
                    avg_ssim.append(ssim)
                    avg_lpips.append(lpips)

                    print('{:6.4f}/{:6.4f}/{:6.4f}'.format(psnr, ssim, lpips), end='\t\t')
            print('')

        avg_psnr = np.mean(avg_psnr)
        avg_ssim = np.mean(avg_ssim)
        avg_lpips = np.mean(avg_lpips)
        psnr_list.append(avg_psnr)
        ssim_list.append(avg_ssim)
        lpips_list.append(avg_lpips)
        time_list.append(time_)
        print('Test PSNR: {:.4f}, SSIM: {:.4f}, LPIPS: {:.4f}, TIME: {:.4f} in {}'.format(avg_psnr, avg_ssim, avg_lpips,
                                                                                          time_, image_name))
        xls_list.append([image_name, avg_psnr, avg_ssim, avg_lpips, time_])

        dir_save_path_tmp = save_path + f'{view_n_in}_{view_n_out}/{image_name}/'
        if not os.path.exists(dir_save_path_tmp):
            os.makedirs(dir_save_path_tmp)

        if save_img:
            for i in range(0, view_n_out):
                for j in range(0, view_n_out):
                    img_save_path = dir_save_path_tmp + str(i).zfill(2) + '_' + str(j).zfill(2) + '.png'
                    hr_rgb_item = Image.fromarray((pred_hr[i, j, :, :] * 255.0).astype(np.uint8))
                    hr_rgb_item.save(img_save_path)

    print('===> Avg. PSNR: {:.4f} dB / Avg. SSIM: {:.4f} / Avg. LPIPS: {:.4f} / Time: {:.6f}'
          .format(np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list), np.mean(time_list)))
    xls_list.append(['average', np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list), np.mean(time_list)])

    xls_list = np.array(xls_list)

    result = pd.DataFrame(xls_list, columns=['image', 'psnr', 'ssim', 'lpips', 'time'])
    save_name = f'result_{view_n_in}_{view_n_out}_{extra}_{datasets_name}.csv'
    result.to_csv(save_path + save_name)


def test_grid(img_path, image_list, model, view_n_out, view_n_in, view_n_oir, extra, position, save_path,
              datasets_name, is_mix, is_multi_channel, save_img):
    cut = 22
    xls_list = []
    if is_mix:
        extra_0_list = [0, 1, 2]
    else:
        extra_0_list = [extra]
    for extra_0 in extra_0_list:
        psnr_list = []
        ssim_list = []
        time_list = []
        print(f'===> Task {view_n_in}x{view_n_in} -> {view_n_out}x{view_n_out} Extra {extra_0}')
        stride_0 = (view_n_out - 1 - 2 * extra_0) // (view_n_in - 1)
        margin_0 = [i * stride_0 + extra_0 for i in range(view_n_in)]
        for index, image_name in enumerate(image_list):
            avg_psnr = []
            avg_ssim = []
            print('[{}/{}]'.format(index + 1, len(image_list)), image_name)

            if datasets_name == 'HCI':
                gt_image = utils.image_prepare_HCI(img_path + image_name, view_n_oir,
                                                       view_n_out, is_multi_channel)
            elif datasets_name == 'HCI_old':
                gt_image = utils.image_prepare_HCI_old(img_path + image_name, view_n_oir,
                                                           view_n_out, is_multi_channel)
            elif datasets_name == 'DLFD':
                gt_image = utils.image_prepare_DLFD(img_path + image_name, view_n_oir,
                                                        view_n_out, is_multi_channel)
            elif datasets_name in ['30scenes', 'occlusions', 'reflective']:
                gt_image = utils.image_prepare_Lytro(img_path + image_name, view_n_oir,
                                                         view_n_out, is_multi_channel)
            if is_multi_channel:
                lr, gt_hr = utils.image_prepare_rgb(gt_image, margin_0, view_n_in)
            else:
                lr, gt_hr, lr_cbcr = utils.image_prepare_ycbcr(gt_image, margin_0, view_n_in)

            pred_hr, time_ = predict(model, lr, position, margin_0)
            if is_multi_channel:
                pred_hr = rearrange(pred_hr, 'b (n c) h w -> b n h w c', c=3)
            pred_hr = pred_hr.cpu().numpy()

            time_list.append(time_)

            for i in range(view_n_out):
                for j in range(view_n_out):
                    if i in margin_0 and j in margin_0:
                        print('      -/-      ', end='\t\t')
                    else:
                        gt_pred_tmp = pred_hr[0, i * view_n_out + j, cut:-cut, cut:-cut]
                        gt_hr_y_tmp = gt_hr[i, j, cut:-cut, cut:-cut]
                        psnr = peak_signal_noise_ratio(gt_hr_y_tmp, gt_pred_tmp)
                        ssim = structural_similarity((gt_pred_tmp * 255.0).astype(np.uint8),
                                                     (gt_hr_y_tmp * 255.0).astype(np.uint8),
                                                     gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                                                     multichannel=is_multi_channel)

                        avg_psnr.append(psnr)
                        avg_ssim.append(ssim)
                        print('{:6.4f}/{:6.4f}'.format(psnr, ssim), end='\t\t')
                print('')

            if save_img:
                dir_save_path_tmp = save_path + f'{view_n_in}{view_n_out}{extra_0}/{image_name}/'
                if not os.path.exists(dir_save_path_tmp):
                    os.makedirs(dir_save_path_tmp)
                if is_multi_channel:
                    for j in range(view_n_out * view_n_out):
                        img_save_path = dir_save_path_tmp + str(j // view_n_out) + str(j % view_n_out) + '.png'
                        cv2.imwrite(img_save_path, pred_hr[0, j, ..., ::-1] * 255.0)
                else:
                    hr_cbcr = predict_cbcr(lr_cbcr, view_n_out)
                    for j in range(view_n_out * view_n_out):
                        hr_y_item = np.clip(pred_hr[0, j, :, :] * 255.0, 16.0, 235.0)
                        hr_y_item = hr_y_item[..., np.newaxis]
                        hr_cbcr_item = hr_cbcr[j // view_n_out, j % view_n_out, :, :, :]
                        hr_ycbcr_item = np.concatenate((hr_y_item, hr_cbcr_item), 2)
                        hr_rgb_item = color.ycbcr2rgb(hr_ycbcr_item)[..., ::-1] * 255.0
                        img_save_path = dir_save_path_tmp + str(j // view_n_out) + str(j % view_n_out) + '.png'
                        cv2.imwrite(img_save_path, hr_rgb_item)
            avg_psnr = np.mean(avg_psnr)
            avg_ssim = np.mean(avg_ssim)
            psnr_list.append(avg_psnr)
            ssim_list.append(avg_ssim)
            time_list.append(time_)
            print('Test PSNR: {:.4f}, SSIM: {:.4f}, TIME: {:.4f} in {}'.format(avg_psnr, avg_ssim, time_, image_name))
            xls_list.append([image_name, avg_psnr, avg_ssim, time_])

        print('===> Avg. PSNR: {:.4f} dB / Avg. SSIM: {:.4f} / Time: {:.6f}'
              .format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))
        xls_list.append(['average', np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)])

    xls_list = np.array(xls_list)

    result = pd.DataFrame(xls_list, columns=['image', 'psnr', 'ssim', 'time'])
    save_name = f'result_{view_n_in}_{view_n_out}_{extra}_{datasets_name}.csv'
    result.to_csv(save_path + save_name)


def test_main(img_path, model_path, view_n_in, view_n_out, extra, datasets, datasets_name, save_path='../result/',
              gpu_no=0, is_save=False, is_multi_channel=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)
    torch.backends.cudnn.benchmark = True

    if extra == -1:
        is_mix = True
        model_name = f'{view_n_in}_{view_n_out}_X_{datasets}'
    else:
        is_mix = False
        model_name = f'{view_n_in}_{view_n_out}_{extra}_{datasets}'

    if is_multi_channel:
        model_name += '_RGB'
    else:
        model_name += '_Y'

    print('=' * 40)
    print('create save directory...')
    save_path += f'{model_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sys.stdout = Logger(save_path + 'test_{}.log'.format(int(time.time())), sys.stdout)
    print('done')
    print('=' * 40)
    print('build network...')

    model = LF_SAPSNR(is_multi_channel=is_multi_channel)

    model.cuda()
    model.eval()
    print('done')
    print('=' * 40)
    print('load model...')

    dir_model = model_path + model_name

    checkpoint = torch.load(dir_model + '.pkl')

    if checkpoint is None:
        print('cannot find model!')
        return
    else:
        model.load_state_dict(checkpoint)

    print('done')
    print('=' * 40)
    print('predict image...')

    image_list = []
    for line in open(f'./list/Test_{datasets_name}.txt'):
        image_list.append(line.strip())
    view_n_ori = int(image_list[0])
    image_list = image_list[1:]

    position = utils.get_position(view_n_out)

    if datasets == 'Stanf':
        test_Stanf_grid(img_path, image_list, model, view_n_out, view_n_in, view_n_ori, extra, position, save_path,
                        datasets_name, is_save)
    else:
        test_grid(img_path, image_list, model, view_n_out, view_n_in, view_n_ori, extra, position, save_path,
                  datasets_name, is_mix, is_multi_channel, is_save)

    print('all done')


def predict(model, lr_y, position, margin_0, max_length=512):
    with torch.no_grad():
        lr_y = lr_y[np.newaxis, :, :, :, :]
        lr_y = torch.from_numpy(lr_y.copy())
        lr_y = lr_y.cuda()

        time_item_start = time.time()
        if lr_y.shape[-1] > max_length or lr_y.shape[-2] > max_length:
            hr_y = utils.overlap_crop_forward(lr_y, position, margin_0, None, model, max_length=max_length, mod=8)
        else:
            hr_y = model(lr_y, position, None, margin_0)
        return hr_y, time.time() - time_item_start


def predict_cbcr(lr_cbcr, view_n_new):
    hr_cbcr = np.zeros((view_n_new, view_n_new, lr_cbcr.shape[2], lr_cbcr.shape[3], 2))
    for i in range(lr_cbcr.shape[2]):
        for j in range(lr_cbcr.shape[3]):
            image_interlinear = cv2.resize(lr_cbcr[:, :, i, j, :], (view_n_new, view_n_new),
                                           interpolation=cv2.INTER_CUBIC)
            hr_cbcr[:, :, i, j, :] = image_interlinear
    return hr_cbcr


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Unsupported value encountered.')


def opts_parser():
    usage = "LF-SAPSNR"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-i', '--img_path', type=str, default=None, dest='img_path',
        help='Loading LF images from this path: (default: %(default)s)')
    parser.add_argument(
        '-m', '--model_path', type=str, default='../pretrain_model/', dest='model_path',
        help='Loading pretrained models from this path: (default: %(default)s)')
    parser.add_argument(
        '-in', '--view_n_in', type=int, default=2, dest='view_n_in',
        help='Angular resolution of input LF images: (default: %(default)s)')
    parser.add_argument(
        '-out', '--view_n_out', type=int, default=7, dest='view_n_out',
        help='Angular resolution of output LF images: (default: %(default)s)')
    parser.add_argument(
        '-e', '--extra', type=int, default=0, dest='extra',
        help='The value of interpolation or extrapolation, set -1 for No-Per-Task: (default: %(default)s)')
    parser.add_argument(
        '-d', '--datasets', type=str, default=None, dest='datasets',
        help='the type of datasets: (default: %(default)s)')
    parser.add_argument(
        '-dn', '--datasets_name', type=str, default=None, dest='datasets_name',
        help='the name of datasets: (default: %(default)s)')
    parser.add_argument(
        '-imc', '--is_multi_channel', type=str2bool, default=False, dest='is_multi_channel',
        help='Whether to process multi-channel (RGB) LF images: (default: %(default)s)')
    parser.add_argument(
        '-is', '--is_save', type=str2bool, default=False, dest='is_save',
        help='Whether to save LF images or not: (default: %(default)s)')
    parser.add_argument(
        '-s', '--save_path', type=str, default='../result/', dest='save_path',
        help='Save LF images to this path: (default: %(default)s)')
    parser.add_argument(
        '-g', '--gpu_no', type=int, default=0, dest='gpu_no',
        help='GPU used: (default: %(default)s)')

    return parser


if __name__ == '__main__':
    parser = opts_parser()
    args = parser.parse_args()

    test_main(
        img_path=args.img_path,
        model_path=args.model_path,
        view_n_in=args.view_n_in,
        view_n_out=args.view_n_out,
        extra=args.extra,
        datasets=args.datasets,
        datasets_name=args.datasets_name,
        save_path=args.save_path,
        gpu_no=args.gpu_no,
        is_save=args.is_save,
        is_multi_channel=args.is_multi_channel,
    )
