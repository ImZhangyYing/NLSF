import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from Unet import UNet
import SimpleITK as sitk
from dataload3 import BraTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def all_case(unet, image_list, patch_size=(160, 160, 32), stride_xy=80, stride_z=32, save_result=True, test_save_path=None):

    for ith, (image, label) in enumerate(image_list):
        image, label = image.to(device), label.to(device)
        c, ww, hh, dd = image.shape
        # print(image.min(), image.max())
        sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

        res_x1_map = np.zeros((1,) + image.shape[1:]).astype(np.float32)
        res_x2_map = np.zeros((1,) + image.shape[1:]).astype(np.float32)
        res_x3_map = np.zeros((1,) + image.shape[1:]).astype(np.float32)
        res_x4_map = np.zeros((1,) + image.shape[1:]).astype(np.float32)
        cnt = np.zeros(image.shape[1:]).astype(np.float32)

        for x in range(0, sx):
            xs = min(stride_xy * x, ww - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd - patch_size[2])
                    test_patch = image[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                    test_patch = test_patch.unsqueeze(0)

                    x1 = test_patch[:, 0, :, :, :].unsqueeze(0)
                    x2 = test_patch[:, 1, :, :, :].unsqueeze(0)
                    x3 = test_patch[:, 2, :, :, :].unsqueeze(0)
                    x4 = test_patch[:, 3, :, :, :].unsqueeze(0)
                    x1g, x1l, res_x1 = unet(x1)
                    x2g, x2l, res_x2 = unet(x2)
                    x3g, x3l, res_x3 = unet(x3)
                    x4g, x4l, res_x4 = unet(x4)

                    res_x1 = res_x1.cpu().data.numpy()
                    res_x1 = res_x1[0, :, :, :, :]
                    res_x2 = res_x2.cpu().data.numpy()
                    res_x2 = res_x2[0, :, :, :, :]
                    res_x3 = res_x3.cpu().data.numpy()
                    res_x3 = res_x3[0, :, :, :, :]
                    res_x4 = res_x4.cpu().data.numpy()
                    res_x4 = res_x4[0, :, :, :, :]


                    res_x1_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = res_x1_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + res_x1
                    res_x2_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = res_x2_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + res_x2
                    res_x3_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = res_x3_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + res_x3
                    res_x4_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = res_x4_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + res_x4

                    cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1

        res_x1_map = res_x1_map / np.expand_dims(cnt, axis=0)
        res_x2_map = res_x2_map / np.expand_dims(cnt, axis=0)
        res_x3_map = res_x3_map / np.expand_dims(cnt, axis=0)
        res_x4_map = res_x4_map / np.expand_dims(cnt, axis=0)
        res_x1_map = res_x1_map.squeeze().transpose(2, 0, 1)
        res_x2_map = res_x2_map.squeeze().transpose(2, 0, 1)
        res_x3_map = res_x3_map.squeeze().transpose(2, 0, 1)
        res_x4_map = res_x4_map.squeeze().transpose(2, 0, 1)

        if save_result:
            savedImg1 = sitk.GetImageFromArray(res_x1_map.astype(np.float32))
            sitk.WriteImage(savedImg1, test_save_path + str(ith) + '-x1.nii.gz')
            savedImg2 = sitk.GetImageFromArray(res_x2_map.astype(np.float32))
            sitk.WriteImage(savedImg2, test_save_path + str(ith) + '-x2.nii.gz')
            savedImg3 = sitk.GetImageFromArray(res_x3_map.astype(np.float32))
            sitk.WriteImage(savedImg3, test_save_path + str(ith) + '-x3.nii.gz')
            savedImg4 = sitk.GetImageFromArray(res_x4_map.astype(np.float32))
            sitk.WriteImage(savedImg4, test_save_path + str(ith) + '-x4.nii.gz')

        print('test over')


if __name__ == '__main__':

    test_save_path = './results/'
    unet = UNet(1).to(device)
    unet.load_state_dict(torch.load('./model/unet-model1.pth'), False)
    train_set = {'root': './Brats_2021_dataset/BraTS2021_Training_Data', 'flist': './Brats_2021_dataset/test.txt'}
    image_list = BraTS(train_set, transform=None)
    with torch.no_grad():
        all_case(unet, image_list, patch_size=(160, 160, 64), stride_xy=160, stride_z=64, save_result=True,test_save_path=test_save_path)




