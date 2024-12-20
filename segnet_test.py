import os
import math
import torch
import torch.nn.functional as F
import numpy as np
# import h5py
# import nibabel as nib
from medpy import metric
from Segnet import SegNet
from dataload import *
from Unet import UNet
from fusenet import FuseNet



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_save_path = './results/'
train_set = {'root': './Brats_2021_dataset/BraTS2021_Training_Data', 'flist': './Brats_2021_dataset/train.txt'}
image_list = BraTS(train_set, transform=None)

#加载训练好的unet
unet = UNet(1).to(device)
unet_path = torch.load('./model/unet-model3.pth')
unet.load_state_dict(unet_path['unet'])
#加载好训练的fusenet
fusenet = FuseNet().to(device)
fuse_parameters = torch.load('./model/fuse-model2.pth')
fusenet.load_state_dict(fuse_parameters['fusenet'])

#加载训练好的seg-net
net = SegNet().to(device)
model_path = './model/model.pth'
weight_dict = torch.load(model_path)
net.load_state_dict(weight_dict['model'])
net.eval()


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd



num_classes = 4
patch_size = (160, 160, 64)
stride_xy = 32
stride_z = 16
save_result = True

with torch.no_grad():
    total_ET_metric = 0.0
    total_TC_metric = 0.0
    total_WT_metric = 0.0
    total_metric = 0
    for ith,(image, label) in enumerate(image_list):
        image, label = image.to(device), label.to(device)
        c, ww, hh, dd = image.shape
        sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
        # print("{}, {}, {}".format(sx, sy, sz))
        score_map = np.zeros((num_classes,) + image.shape[1:]).astype(np.float32)
        fused_map = np.zeros((1,) + image.shape[1:]).astype(np.float32)
        cnt = np.zeros(image.shape[1:]).astype(np.float32)

        for x in range(0, sx):
            xs = min(stride_xy * x, ww - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd - patch_size[2])
                    test_patch = image[:, xs:xs + patch_size[0], ys:ys + patch_size[1],zs:zs + patch_size[2]].unsqueeze(0)

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



                    y1 = net(x1l, x2l, x3l, x4l,fused_image)
                    y = F.softmax(y1, dim=1)
                    y = y.cpu().data.numpy()
                    y = y[0, :, :, :, :]
                    fused = fused_image.cpu().data.numpy()
                    fused = fused[0, :, :, :, :]

                    score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                    fused_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = fused_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + fused
                    cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
        score_map = score_map / np.expand_dims(cnt, axis=0)
        fused = fused_map / np.expand_dims(cnt, axis=0)
        label_map = np.argmax(score_map, axis=0)
        fused = fused.squeeze()

        prediction = label_map
        label = label.cpu().data.numpy()
        print(np.unique(prediction),np.unique(label))

        if save_result:
            # 预测标签
            pred = sitk.GetImageFromArray(prediction.transpose(2, 0, 1).astype(np.float32))
            sitk.WriteImage(pred, test_save_path + str(ith) + '-pred.nii.gz')
            # 真实标签
            gt = sitk.GetImageFromArray(label[:].transpose(2, 0, 1).astype(np.float32))
            sitk.WriteImage(gt, test_save_path + str(ith) + '-gt.nii.gz')
            # 融合图像
            savedImgf = sitk.GetImageFromArray(fused.transpose(2, 0, 1).astype(np.float32))
            sitk.WriteImage(savedImgf, test_save_path + str(ith) + 'fused.nii.gz')
            # nib.save(nib.Nifti1Image(fused.astype(np.float32), np.eye(4)), test_save_path + str(ith) + 'fused.nii.gz')
            # nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),test_save_path + str(ith)+'fuse-pred.nii.gz')
            # nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + str(ith)+ 'gt.nii.gz')
            # 融合图像
    #     if np.sum(prediction)==0:
    #         ET_metrics = (0,0,0,0)
    #         TC_metrics = (0, 0, 0, 0)
    #         WT_metrics = (0, 0, 0, 0)
    #     else:
    #         ET_metrics = brats_metrics(prediction, label)
    #         TC_metrics = brats_metrics(prediction, label)
    #         WT_metrics = brats_metrics(prediction, label)
    #         # ET_metrics = brats_metrics(prediction, label, 'ET')
    #         # TC_metrics = brats_metrics(prediction, label, 'TC')
    #         # WT_metrics = brats_metrics(prediction, label, 'WT')
    #
    #     print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, ET_metrics[0], ET_metrics[1], ET_metrics[2], ET_metrics[3]))
    #     print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, TC_metrics[0], TC_metrics[1],TC_metrics[2], TC_metrics[3]))
    #     print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, WT_metrics[0], WT_metrics[1],WT_metrics[2], WT_metrics[3]))
    #     total_ET_metric += np.asarray(ET_metrics)
    #     total_TC_metric += np.asarray(TC_metrics)
    #     total_WT_metric += np.asarray(WT_metrics)
    #
    #
    # # avg_metric = total_metric / len(image_list)
    # # print('average metric is {}'.format(avg_metric))
    #
    # avg_ET_metric = total_ET_metric / len(image_list)
    # avg_TC_metric = total_TC_metric / len(image_list)
    # avg_WT_metric = total_WT_metric / len(image_list)
    # print('average ET metric is {}'.format(avg_ET_metric))
    # print('average TC metric is {}'.format(avg_TC_metric))
    # print('average WT metric is {}'.format(avg_WT_metric))












