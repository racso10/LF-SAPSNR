import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import math

from modules.deform_conv import DeformConv


class ORI_CNN(nn.Module):
    def __init__(self, channel, is_multi_channel=False):
        super(ORI_CNN, self).__init__()
        if is_multi_channel:
            in_channels = 3
        else:
            in_channels = 1
        self.conv = nn.Conv3d(in_channels, channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1),
                              bias=True)
        self.PRelu = nn.PReLU()
        self.channel = channel
        self.is_multi_channel = is_multi_channel

    def forward(self, x):
        if self.is_multi_channel:
            batch, height_view, width_view, height, width, channel = list(x.shape)
            x = rearrange(x, 'b u v h w c -> b c (u v) h w')
        else:
            batch, height_view, width_view, height, width = list(x.shape)
            x = torch.unsqueeze(x, 1)
            x = x.reshape(batch, 1, height_view * width_view, height, width)
        x = self.PRelu(self.conv(x))
        x = x.reshape(batch, self.channel, height_view, width_view, height, width)
        return x


class Composite_Feature_Extraction(nn.Module):
    def __init__(self, block_num=4, channel=32, out_channels=1):
        super(Composite_Feature_Extraction, self).__init__()
        layers = list()
        for _ in range(block_num):
            layers.append(SA_ASPP(channel=channel))
        self.sa_aspp = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(channel, out_channels, kernel_size=3, padding=1)
        self.out_channels = out_channels

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = list(x.shape)
        x = self.sa_aspp(x)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(batch * height_view * width_view, channel, height, width)
        x = self.final_conv(x)
        x = x.reshape(batch, height_view, width_view, self.out_channels, height, width).permute(0, 3, 1, 2, 4, 5)
        return x


class Coordinates_Generation(nn.Module):
    def __init__(self, channel, num_sampling):
        super(Coordinates_Generation, self).__init__()
        add_channels = 2
        self.coord_cnn = nn.Sequential(
            nn.Conv2d(channel + add_channels, channel, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1),
            nn.PReLU())
        out_channels = num_sampling
        self.final_cnn = nn.Conv2d(channel, out_channels, kernel_size=3, padding=1)

    def forward(self, x, view_u, view_v, i, j, margin):
        B, C, X, Y = list(x.shape)

        pos_dist_u = (view_u - margin[i]) / (margin[-1] - margin[0])
        pos_dist_v = (view_v - margin[j]) / (margin[-1] - margin[0])

        pos_dist = torch.cat((pos_dist_u.unsqueeze(1), pos_dist_v.unsqueeze(1)), dim=1).unsqueeze(-1).unsqueeze(
            -1).cuda()

        pos_dist = pos_dist.expand(-1, -1, X, Y)
        x = torch.cat((x, pos_dist), dim=1)
        x = self.coord_cnn(x)
        x = self.final_cnn(x)
        return x


class Feature_Fusion(nn.Module):
    def __init__(self, channel):
        super(Feature_Fusion, self).__init__()
        layers = list()
        while channel != 1:
            layers.append(nn.Conv3d(channel, channel // 4, kernel_size=(3, 3, 1), padding=(1, 1, 0)))
            layers.append(nn.PReLU())
            channel //= 4
        self.seqn = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = list(x.shape)
        x = x.reshape(batch, channel, height_view, width_view, height * width)
        weight = self.seqn(x)
        weight = weight.reshape(batch, 1, height_view * width_view, height, width)
        weight = self.softmax(weight)

        x = x.reshape(batch, channel, height_view * width_view, height, width)
        x = torch.mul(x, weight)
        x = torch.sum(x, dim=2)

        return x


class Structure_Aware_Pre_Selected_Rendering(nn.Module):
    def __init__(self, channel, num_sampling):
        super(Structure_Aware_Pre_Selected_Rendering, self).__init__()

        self.coord_generation = Coordinates_Generation(channel, num_sampling)

        self.pre_selected_rendering = DeformConv(channel, channel, kernel_size=int(math.sqrt(num_sampling)), stride=1,
                                                 padding=(int(math.sqrt(num_sampling)) - 1) // 2, im2col_step=128)

        self.PRelu0 = nn.PReLU()

        self.num_sampling = num_sampling
        self.p_n = self._get_p_n()

    def _get_p_n(self):
        kernel_size = int(math.sqrt(self.num_sampling))
        p_n_x, p_n_y = np.meshgrid(range(-(kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1),
                                   range(-(kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1), indexing='ij')
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2 * self.num_sampling, 1, 1))
        p_n = torch.from_numpy(p_n).float().cuda().detach()
        return p_n

    def forward(self, x, view_u, view_v, margin):
        batch, channel, height_view, width_view, height, width = list(x.shape)
        for i in range(height_view):
            for j in range(width_view):
                samping_coordinates = self.coord_generation(x[:, :, i, j, :, :], view_u, view_v, i, j, margin)

                # Calculate the slope of the epipolar line
                dist_u = view_u - margin[i]
                dist_v = view_v - margin[j]

                # Normalize the horizontal and vertical components
                pos_dist_u = dist_u / torch.sqrt(dist_u * dist_u + dist_v * dist_v + 1e-8)
                pos_dist_v = dist_v / torch.sqrt(dist_u * dist_u + dist_v * dist_v + 1e-8)

                pos_dist_u = pos_dist_u.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
                pos_dist_v = pos_dist_v.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
                samping_coordinates = torch.cat((samping_coordinates * pos_dist_u, samping_coordinates * pos_dist_v),
                                                dim=1)

                # Remove the regular grid offsets in [J. Dai, H. Qi, Y. Xiong, Y. Li, Y. Wei,
                # Deformable convolutional networks, in: Proc. of ICCV, 2017.]
                samping_coordinates = samping_coordinates - self.p_n

                samping_coordinates = rearrange(samping_coordinates, 'b (a N) h w -> b (N a) h w', N=self.num_sampling)

                # Use the deformable convolution to sample the corresponding points
                # of the epipolar line on the source view feature according to the
                # offsets to obtain the pre-selected epipolar features
                x[:, :, i, j, :, :] = self.PRelu0(
                    self.pre_selected_rendering(x[:, :, i, j, :, :].contiguous(), samping_coordinates))

        return x


class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(channel, 64, kernel_size=1, bias=False),
                                nn.PReLU(),
                                nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                nn.PReLU(),
                                nn.Conv2d(64, channel, kernel_size=1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        x = x * (1 + self.sigmoid(out))
        return x


class SA_ASPP(nn.Module):
    def __init__(self, channel=32):
        super(SA_ASPP, self).__init__()
        self.s_d1_conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // 4, kernel_size=3, padding=1, stride=1,
                      dilation=1), nn.PReLU())
        self.s_d2_conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // 4, kernel_size=3, padding=2, stride=1,
                      dilation=2), nn.PReLU())
        self.s_d4_conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // 4, kernel_size=3, padding=4, stride=1,
                      dilation=4), nn.PReLU())
        self.s_d8_conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // 4, kernel_size=3, padding=8, stride=1,
                      dilation=8), nn.PReLU())
        self.s_fusion = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1,
                      dilation=1), nn.PReLU())
        self.a_conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1, stride=1), nn.PReLU())

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = list(x.shape)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(batch * height_view * width_view, channel, height, width)
        x_all = []
        x_all.append(self.s_d1_conv(x))
        x_all.append(self.s_d2_conv(x))
        x_all.append(self.s_d4_conv(x))
        x_all.append(self.s_d8_conv(x))
        x_all = self.s_fusion(torch.cat(x_all, dim=1))
        x_all = x_all.reshape(batch, height_view, width_view, channel, height, width).permute(0, 4, 5, 3, 1, 2).reshape(
            batch * height * width, channel, height_view, width_view)
        x_all = self.a_conv(x_all)
        x_all = x_all.reshape(batch, height, width, channel, height_view, width_view).permute(0, 3, 4, 5, 1, 2)
        return x_all


class res_block_2d(nn.Module):
    def __init__(self, channel):
        super(res_block_2d, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.PRelu = nn.PReLU()

    def forward(self, x):
        x = x + self.PRelu(self.conv(x))
        return x


class Color_Encoding(nn.Module):
    def __init__(self, block_num, channel, is_multi_channel=False):
        super(Color_Encoding, self).__init__()
        layers = list()
        for _ in range(block_num):
            layers.append(res_block_2d(channel))
        self.res_cnn = nn.Sequential(*layers)
        if is_multi_channel:
            out_channels = 3
        else:
            out_channels = 1
        self.final_cnn = nn.Conv2d(channel, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.res_cnn(x)
        x = self.final_cnn(x)
        return x


class LF_SAPSNR(nn.Module):
    def __init__(self, FE_block_num=3, res_block_num=3, num_sampling=9, channel=64, is_multi_channel=False):
        super(LF_SAPSNR, self).__init__()
        self.ori_cnn = ORI_CNN(channel, is_multi_channel=is_multi_channel)
        self.fea_part = Composite_Feature_Extraction(block_num=FE_block_num, channel=channel, out_channels=channel)

        self.render_part = Structure_Aware_Pre_Selected_Rendering(channel, num_sampling)

        self.fusion_part = Feature_Fusion(channel=channel)

        self.encoding_part = Color_Encoding(res_block_num, channel, is_multi_channel=is_multi_channel)

    def forward(self, x, view_u, view_v, margin, test_batch_size=10):
        x = self.ori_cnn(x)
        x = self.fea_part(x)

        if self.training:
            x = self.render_part(x, view_u, view_v, margin)
            x = self.fusion_part(x)
            x = self.encoding_part(x)
        else:
            output = []
            for i in range(0, view_u.shape[1], test_batch_size):
                if i + test_batch_size > view_u.shape[1]:
                    j = view_u.shape[1]
                else:
                    j = i + test_batch_size
                view_uu = view_u[0, i:j]
                view_vv = view_u[1, i:j]
                x_tmp = self.render_part(x.expand(j - i, -1, -1, -1, -1, -1).clone(), view_uu, view_vv, margin)
                x_tmp = self.fusion_part(x_tmp)
                x_tmp = self.encoding_part(x_tmp)
                x_tmp = rearrange(x_tmp, 'b c x y -> (b c) x y').unsqueeze(0)
                output.append(x_tmp)
            x = torch.cat(output, dim=1)

        return x
