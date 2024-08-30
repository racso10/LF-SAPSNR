import torch
import time
import os
from einops import rearrange
from argparse import ArgumentParser, ArgumentTypeError

from model import LF_SAPSNR
from dataset import ASRLF_Dataset
import utils

MAX_EPOCH = 10001


def train_main(dataset, is_multi_channel, img_path, view_n_in, view_n_out, extra, batch_size, crop_size, base_lr,
               pretrain_model, current_epoch, cuda_no):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_no)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    ''' Define Model(set parameters)'''
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    criterion_train = torch.nn.L1Loss()

    model = LF_SAPSNR(is_multi_channel=is_multi_channel)

    utils.get_parameter_number(model)

    model.apply(utils.weights_init_xavier)

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=3000, gamma=0.5)

    if extra == -1:
        task_name = f'{view_n_in}_{view_n_out}_X_{dataset}'
        is_mix = True
    else:
        task_name = f'{view_n_in}_{view_n_out}_{extra}_{dataset}'
        is_mix = False

    if is_multi_channel:
        task_name += '_RGB'
    else:
        task_name += '_Y'

    if dataset == 'Syn':
        is_shear = True
        repeat_size = 128
    elif dataset == 'Lytro':
        is_shear = False
        repeat_size = 64

    if pretrain_model is not None:
        checkpoint = torch.load(f'../net_store/{pretrain_model}/{str(current_epoch).zfill(5)}.pkl')

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    dir_save_name = f'../net_store/{task_name}'

    print('model save path:', dir_save_name)
    if not os.path.exists(dir_save_name):
        os.makedirs(dir_save_name)

    stride = (view_n_out - 1 - 2 * extra) // (view_n_in - 1)
    margin = [i * stride + extra for i in range(view_n_in)]

    train_dataset = ASRLF_Dataset(img_path, dataset, view_n_in, view_n_out, margin,
                                  repeat_size=repeat_size, crop_size=crop_size, is_flip=True, is_rotation=True,
                                  is_shear=is_shear, is_mix=is_mix, is_multi_channel=is_multi_channel)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(current_epoch, MAX_EPOCH):
        if epoch % 100 == 0:
            state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict()}
            torch.save(state, dir_save_name + f'/{str(epoch).zfill(5)}.pkl')
            torch.save(model.state_dict(), f'../net_store/{task_name}.pkl')

        train_epoch(train_loader, model, epoch, criterion_train, optimizer, is_multi_channel)
        scheduler.step()


def train_epoch(train_loader, model, epoch, criterion, optimizer, is_multi_channel):
    time_start = time.time()
    model.train()

    total_loss = 0
    count = 0

    for i, (train_data, gt_data, view_u_list, view_v_list, margin_0) in enumerate(train_loader):
        train_data, gt_data = train_data.cuda(), gt_data.cuda()
        optimizer.zero_grad()
        gt_pred = model(train_data, view_u_list, view_v_list, margin_0)

        if is_multi_channel:
            gt_data = rearrange(gt_data, 'b u v h w c -> b (u v c) h w')
            loss = criterion(gt_pred, gt_data)
        else:
            loss = criterion(gt_pred[:, 0, :, :], gt_data[:, 0, 0, :, :])
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        count += 1

    time_end = time.time()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    print('Train Epoch: {} Learning rate: {:.2e} Time: {:.2f}s Average Loss: {:.6f} '
          .format(epoch, lr, time_end - time_start, total_loss / count))


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
        '-d', '--datasets', type=str, default=None, dest='datasets',
        help='the type of datasets: (default: %(default)s)')
    parser.add_argument(
        '-imc', '--is_multi_channel', type=str2bool, default=False, dest='is_multi_channel',
        help='Whether to process multi-channel (RGB) LF images: (default: %(default)s)')
    parser.add_argument(
        '-i', '--img_path', type=str, default=None, dest='img_path',
        help='Loading LF images from this path: (default: %(default)s)')
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
        '-b', '--batch_size', type=int, default=32, dest='batch_size',
        help='Batch size of training: (default: %(default)s)')
    parser.add_argument(
        '-c', '--crop_size', type=int, default=32, dest='crop_size',
        help='Crop size of training: (default: %(default)s)')
    parser.add_argument(
        '-lr', '--base_lr', type=float, default=0.001, dest='base_lr',
        help='Crop size of training: (default: %(default)s)')
    parser.add_argument(
        '-pm', '--pretrain_model', type=str, default=None, dest='pretrain_model',
        help='The name of pertrained model: (default: %(default)s)')
    parser.add_argument(
        '-ce', '--current_epoch', type=int, default=0, dest='current_epoch',
        help='The training epoch of the pertrained model: (default: %(default)s)')
    parser.add_argument(
        '-g', '--gpu_no', type=int, default=0, dest='gpu_no',
        help='GPU used: (default: %(default)s)')

    return parser


if __name__ == '__main__':
    parser = opts_parser()
    args = parser.parse_args()

    train_main(
        dataset=args.datasets,
        is_multi_channel=args.is_multi_channel,
        img_path=args.img_path,
        view_n_in=args.view_n_in,
        view_n_out=args.view_n_out,
        extra=args.extra,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        base_lr=args.base_lr,
        pretrain_model=args.pretrain_model,
        current_epoch=args.current_epoch,
        cuda_no=args.gpu_no,
    )
