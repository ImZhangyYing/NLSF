import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import nibabel as nib
from medpy import metric
from Unet import UNet
from fusenet import FuseNet
import SimpleITK as sitk
from dataload import BraTS


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_save_path = './results/'
train_set = {'root': './Brats_2021_dataset/BraTS2021_Training_Data', 'flist': './Brats_2021_dataset/test.txt'}
image_list = BraTS(train_set, transform=None)

#加载训练好的unet
unet = UNet(1).to(device)
unet_path = torch.load('./model/unet-model2.pth')
unet.load_state_dict(unet_path['model'], strict=False)

#加载好训练的fusenet
fusenet = FuseNet().to(device)
fuse_parameters = torch.load('./model/fuse-model1.pth')
fusenet.load_state_dict(fuse_parameters['model'])
fusenet.eval()


num_classes = 4
patch_size = (160, 160, 64)
stride_xy = 32
stride_z = 16
save_result = True

with torch.no_grad():
    for ith, (image, label) in enumerate(image_list):
        image, label = image.to(device), label.to(device)

        c, ww, hh, dd = image.shape
        sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
        # math.ceil()向上取整；math.floor()向下取整

        score_map = np.zeros((num_classes,) + image.shape[1:]).astype(np.float32)
        fused_map = np.zeros((1,) + image.shape[1:]).astype(np.float32)
        cnt = np.zeros(image.shape[1:]).astype(np.float32)

        for x in range(0, sx):
            xs = min(stride_xy * x, ww - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd - patch_size[2])
                    test_patch = image[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]].unsqueeze(0)

                    x1 = test_patch[:, 0, :, :, :].unsqueeze(0)
                    x2 = test_patch[:, 1, :, :, :].unsqueeze(0)
                    x3 = test_patch[:, 2, :, :, :].unsqueeze(0)
                    x4 = test_patch[:, 3, :, :, :].unsqueeze(0)
                    x1g, x1l, res_x1 = unet(x1)
                    x2g, x2l, res_x2 = unet(x2)
                    x3g, x3l, res_x3 = unet(x3)
                    x4g, x4l, res_x4 = unet(x4)
                    wA, wB, wC, wD = fusenet(x1g, x2g, x3g, x4g, x1l, x2l, x3l, x4l)
                    fused_image = wA * x1 + wB * x2 + wC * x3 + wD * x4
                    fused = fused_image.cpu().data.numpy()
                    fused = fused[0, :, :, :, :]

                    fused_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = fused_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + fused
                    cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1

        fused = fused_map / np.expand_dims(cnt, axis=0)
        fused = fused.squeeze()

        # 融合图像
        savedImgf = sitk.GetImageFromArray(fused.transpose(2, 0, 1).astype(np.float32))
        sitk.WriteImage(savedImgf, test_save_path + str(ith) + '-fused.nii.gz')
        print('test over')



