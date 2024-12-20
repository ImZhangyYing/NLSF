import torch
import torch.nn as nn
import torch.nn.functional as F

from STN import SpatialTransformer, AffineTransformer
import numpy as np
from Transform_self import SpatialTransform, AppearanceTransform

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//8, out_channels),

            nn.ReLU(),
            # nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.GroupNorm(out_channels//8, out_channels),
            # nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)



class DoubleConvK1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class GVSL(nn.Module):
    def __init__(self, n_channels=1, chan=(32, 64, 128, 64, 32)):
        super(GVSL, self).__init__()
        # self.unet = UNet_base(n_channels=1, cha=chan)
        self.f_conv = DoubleConv(1280, 256)
        # self.f_conv = DoubleConv(640, 256)
        self.sp_conv = DoubleConv(160, 16)

        self.res_conv = nn.Sequential(nn.Conv3d(32, 16, 3, padding=1),
                                      nn.GroupNorm(16//4, 16),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv3d(16, 1, 1))

        self.out_flow = nn.Conv3d(16, 3, 3, padding=1)
        self.fc_rot = nn.Linear(256, 3)
        self.softmax = nn.Softmax(1)
        self.fc_scl = nn.Linear(256, 3)
        self.fc_trans = nn.Linear(256, 3)
        self.fc_shear = nn.Linear(256, 6)

        self.gap = nn.AdaptiveAvgPool3d(1)

        self.atn = AffineTransformer()
        self.stn = SpatialTransformer()
        # self.fusenet = FuseNet()




    def get_affine_mat(self, rot, scale, translate, shear):#求取affine matrix
        # 从旋转参数中提取各轴的旋转角度
        theta_x = rot[:, 0]
        theta_y = rot[:, 1]
        theta_z = rot[:, 2]
        # 从缩放参数中提取各轴的缩放因子
        scale_x = scale[:, 0]
        scale_y = scale[:, 1]
        scale_z = scale[:, 2]
        # 从平移参数中提取各轴的平移量
        trans_x = translate[:, 0]
        trans_y = translate[:, 1]
        trans_z = translate[:, 2]
        # 从剪切参数中提取各轴的剪切角度
        shear_xy = shear[:, 0]
        shear_xz = shear[:, 1]
        shear_yx = shear[:, 2]
        shear_yz = shear[:, 3]
        shear_zx = shear[:, 4]
        shear_zy = shear[:, 5]
        # 构建绕 X 轴的旋转矩阵
        rot_mat_x = torch.FloatTensor([[1, 0, 0], [0, torch.cos(theta_x), -torch.sin(theta_x)],
                                       [0, torch.sin(theta_x), torch.cos(theta_x)]]).cuda()
        rot_mat_x = rot_mat_x[np.newaxis, :, :]
        # 构建绕 Y 轴的旋转矩阵
        rot_mat_y = torch.FloatTensor([[torch.cos(theta_y), 0, torch.sin(theta_y)], [0, 1, 0],
                                       [-torch.sin(theta_y), 0, torch.cos(theta_y)]]).cuda()
        rot_mat_y = rot_mat_y[np.newaxis, :, :]
        # 构建绕 Z 轴的旋转矩阵
        rot_mat_z = torch.FloatTensor(
            [[torch.cos(theta_z), -torch.sin(theta_z), 0], [torch.sin(theta_z), torch.cos(theta_z), 0],
             [0, 0, 1]]).cuda()
        rot_mat_z = rot_mat_z[np.newaxis, :, :]
        # 构建缩放矩阵
        scale_mat = torch.FloatTensor([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, scale_z]]).cuda()
        scale_mat = scale_mat[np.newaxis, :, :]
        # 构建剪切矩阵
        shear_mat = torch.FloatTensor(
            [[1, torch.tan(shear_xy), torch.tan(shear_xz)], [torch.tan(shear_yx), 1, torch.tan(shear_yz)],
             [torch.tan(shear_zx), torch.tan(shear_zy), 1]]).cuda()
        # 构建平移向量
        trans = torch.FloatTensor([trans_x, trans_y, trans_z]).cuda()
        trans = trans[np.newaxis, :, np.newaxis]
        # 计算仿射变换矩阵
        mat = torch.matmul(shear_mat,
                           torch.matmul(scale_mat, torch.matmul(rot_mat_z, torch.matmul(rot_mat_y, rot_mat_x))))
        mat = torch.cat([mat, trans], dim=-1)
        return mat


    # def Affine(self, m, f): # Affine transformation
    #     x = torch.cat([m, f], dim=1)
    #     x = self.f_conv(x)
    #     xcor = self.gap(x).flatten(start_dim=1, end_dim=4)
    #     rot = self.fc_rot(xcor)
    #     scl = self.fc_scl(xcor)
    #     trans = self.fc_trans(xcor)
    #     shear = self.fc_shear(xcor)
    #
    #     rot = torch.clamp(rot, -1, 1) * (np.pi / 9)
    #     scl = torch.clamp(scl, -1, 1) * 0.25 + 1
    #     shear = torch.clamp(shear, -1, 1) * (np.pi / 18)
    #
    #     mat = self.get_affine_mat(rot, scl, trans, shear)
    #     return mat
    def Affine(self, x1, x2, x3, x4, fused): # Affine transformation
        # x1_a = x1.cpu().data.numpy().copy()
        # x1_aug = self.style_aug.rand_aug(x1_a)
        x = torch.cat([x1, x2, x3, x4, fused], dim=1)
        x = self.f_conv(x)
        xcor = self.gap(x).flatten(start_dim=1, end_dim=4)
        rot = self.fc_rot(xcor)
        scl = self.fc_scl(xcor)
        trans = self.fc_trans(xcor)
        shear = self.fc_shear(xcor)

        rot = torch.clamp(rot, -1, 1) * (np.pi / 9)
        scl = torch.clamp(scl, -1, 1) * 0.25 + 1
        shear = torch.clamp(shear, -1, 1) * (np.pi / 18)

        mat = self.get_affine_mat(rot, scl, trans, shear)
        return mat

    def Spatial(self, x1, x2, x3, x4, fused):    # Deformable transformation
        x = torch.cat([x1, x2, x3, x4, fused], dim=1)
        sp_cor = self.sp_conv(x)
        flow = self.out_flow(sp_cor)
        return flow

    def forward(self, features, fused_image):
        [fA_g, fB_g, fC_g, fD_g, fA_l, fB_l, fC_l, fD_l, fused_g, fused_l] = features
        # [features1, features2,features3,features4,featuresf] = features
        # [fA_g, A_g3, A_g2, A_g1, fA_l] = features1
        # [fB_g, B_g3, B_g2, B_g1, fB_l] = features2
        # [fC_g, C_g3, C_g2, C_g1, fC_l] = features3
        # [fD_g, D_g3, D_g2, D_g1, fD_l] = features4
        # [fused_g, D_g3, D_g2, D_g1, fused_l] = featuresf
        # fused_image = fused_image.permute(0, 1, 4, 3, 2)

        # Affine
        # aff_mat_BA = self.Affine(fB_g, fA_g)#affine matrix(Gab=Gba)
        aff_mat = self.Affine(fA_g, fB_g, fC_g, fD_g, fused_g)  # affine matrix(G)
        aff_fuse = self.atn(fused_l, aff_mat)
        # aff_fB_l = self.atn(fB_l, aff_mat)  # lB affine transform(fb_l_Gba)
        # aff_fC_l = self.atn(fC_l, aff_mat)  # lB affine transform(fb_l_Gba)
        # aff_fD_l = self.atn(fD_l, aff_mat)  # lB affine transform(fb_l_Gba)

        # defore
        flow = self.Spatial(aff_fuse, fA_l, fB_l, fC_l, fD_l)  # deformable map (Lab=Lba=DVF)
        # flowB = self.Spatial(aff_fB_l, fA_l, fC_l, fD_l)
        # flowC = self.Spatial(aff_fC_l, fA_l, fB_l, fD_l)
        # flowD = self.Spatial(aff_fD_l, fA_l, fB_l, fC_l)

        # registration
        # aff_fb_g = self.atn(B, aff_mat)  # Gb_ab =affine transform(b,Gab)
        warp_fuse = self.stn(self.atn(fused_image, aff_mat), flow)
        # warp_B = self.stn(self.atn(B, aff_mat), flowB)  # B_ab=spatial transform(Gb_ab, Lab=DVF)
        # # aff_fC_g = self.atn(C, aff_mat)  # Gb_ab =affine transform(b,Gab)
        # warp_C = self.stn(self.atn(C, aff_mat), flowC)  # B_ab=spatial transform(Gb_ab, Lab=DVF)
        # # aff_fD_g = self.atn(D, aff_mat)  # Gb_ab =affine transform(b,Gab)
        # warp_D = self.stn(self.atn(D, aff_mat), flowD)  # B_ab=spatial transform(Gb_ab, Lab=DVF)


        # warp_fuse = warp_fuse.permute(0, 1, 4, 3, 2)
        # fused_image = fused_image.permute(0, 1, 4, 3, 2)



        return warp_fuse, aff_mat, flow
