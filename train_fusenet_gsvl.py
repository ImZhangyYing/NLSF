import os
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 使用第三块显卡
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.transforms import transforms
from dataload import *
from LOSS import cal_dice, cosine_scheduler
from gvsl import GVSL
from gvsl_losses import gradient_loss, ncc_loss, MSE

from Unet import UNet
from fusenet import FuseNet

from pytorch_ssim_3D_master import pytorch_ssim

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--seed', type=int, default=21)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--min_lr', type=float, default=0.00001)
parser.add_argument('--data_path', type=str, default='E:/medical6/Ying/Brats_2021_dataset/BraTS2021_Training_Data')
parser.add_argument('--train_txt', type=str, default='E:/medical6/Ying/Brats_2021_dataset/train.txt')
parser.add_argument('--valid_txt', type=str, default='E:/medical6/Ying/Brats_2021_dataset/valid.txt')
parser.add_argument('--test_txt', type=str, default='E:/medical6/Ying/Brats_2021_dataset/test.txt')
parser.add_argument('--train_log', type=str, default='E:/medical6/Ying/ablation-study/4/model/fuse-model.txt')
parser.add_argument('--weights', type=str, default='E:/medical6/Ying/ablation-study/4/model/unet-model1.pth')
parser.add_argument('--save_unet_path', type=str, default='E:/medical6/Ying/ablation-study/4/model/unet-model2.pth')
parser.add_argument('--save_fusenet_path', type=str, default='E:/medical6/Ying/ablation-study/4/model/fuse-model1.pth')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = UNet(1).to(device)
unet_parameters = torch.load(args.weights)
unet.load_state_dict(unet_parameters['model'])

fusenet = FuseNet().to(device)
gvsl = GVSL().to(device)
optimizer1 = optim.Adam(unet.parameters(), lr=1e-5, weight_decay=5e-4)
optimizer2 = optim.Adam(fusenet.parameters(), lr=1e-4, weight_decay=5e-4)
optimizer3 = optim.SGD(gvsl.parameters(), momentum=0.9, lr=1e-4, weight_decay=5e-4)
ssim_loss = pytorch_ssim.SSIM3D(window_size=11).to(device)

L_smooth = gradient_loss
L_cc = ncc_loss
lp = 5
sp = 5

def train_loop(train_loader,scheduler,epoch):
    fusenet.train()
    running_loss = 0
    pbar = tqdm(train_loader)
    for it, (images, masks) in enumerate(pbar):
        it = len(train_loader) * epoch + it
        param_group = optimizer2.param_groups[0]
        param_group['lr'] = scheduler[it]

        images, masks = images.to(device), masks.to(device)

        x1 = images[:, 0, :, :, :].unsqueeze(0)
        x2 = images[:, 1, :, :, :].unsqueeze(0)
        x3 = images[:, 2, :, :, :].unsqueeze(0)
        x4 = images[:, 3, :, :, :].unsqueeze(0)

        x1g, x1l, res_x1 = unet(x1)
        x2g, x2l, res_x2 = unet(x2)
        x3g, x3l, res_x3 = unet(x3)
        x4g, x4l, res_x4 = unet(x4)

        wA, wB, wC, wD = fusenet(x1g,x2g,x3g,x4g,x1l,x2l,x3l,x4l)
        fused_image = wA*x1 + wB*x2 + wC*x3 + wD*x4
        # fused_image = fusenet(x1g, x2g, x3g, x4g, x1l, x2l, x3l, x4l)

        fg, fl, res_f = unet(fused_image)
        features = [x1g,x2g,x3g,x4g,x1l,x2l,x3l,x4l,fg,fl]
        warp_fuse, aff_mat, flow = gvsl(features, fused_image)
        loss_ncc = (1 / 4) * (L_cc(warp_fuse, x1) + L_cc(warp_fuse, x2) + L_cc(warp_fuse, x3) + L_cc(warp_fuse, x4))
        loss_smooth = L_smooth(flow)
        loss_mse = (1 / 4) * (torch.norm((fused_image - x1), 2) + torch.norm((fused_image - x2), 2) + torch.norm((fused_image - x3), 2) + torch.norm((fused_image - x4), 2))
        ss_loss = (1 / 4) *(ssim_loss(fused_image, x1) + ssim_loss(fused_image, x2) + ssim_loss(fused_image, x3) + ssim_loss(fused_image, x4))
        [u1,index] = torch.max(images,dim=1)
        u1 = u1.unsqueeze(0)

        light_loss = (1 / (80 * 80 * 64)) * (((fused_image - u1).norm(2)))
        Loss = (1 / (80 * 80 * 64)) * loss_mse + loss_ncc + loss_smooth + lp*light_loss - sp*ss_loss

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        Loss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        running_loss += Loss.item()
    loss = running_loss / len(train_loader)

    return {'loss': loss}


def val_loop(val_loader):
    fusenet.eval()
    running_loss = 0
    pbar = tqdm(val_loader)
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            x1 = images[:, 0, :, :, :].unsqueeze(0)
            x2 = images[:, 1, :, :, :].unsqueeze(0)
            x3 = images[:, 2, :, :, :].unsqueeze(0)
            x4 = images[:, 3, :, :, :].unsqueeze(0)

            x1g, x1l, res_x1 = unet(x1)
            x2g, x2l, res_x2 = unet(x2)
            x3g, x3l, res_x3 = unet(x3)
            x4g, x4l, res_x4 = unet(x4)
            wA, wB, wC, wD = fusenet(x1g, x2g, x3g, x4g, x1l, x2l, x3l, x4l)
            fused_image = wA * x1 + wB * x2 + wC * x3 + wD * x4
            # fused_image = fusenet(x1g, x2g, x3g, x4g, x1l, x2l, x3l, x4l)

            fg, fl, res_f = unet(fused_image)
            features = [x1g, x2g, x3g, x4g, x1l, x2l, x3l, x4l, fg, fl]

            warp_fuse, aff_mat, flow = gvsl(features, fused_image)
            loss_ncc = (1 / 4) * (L_cc(warp_fuse, x1) + L_cc(warp_fuse, x2) + L_cc(warp_fuse, x3) + L_cc(warp_fuse, x4))
            loss_smooth = L_smooth(flow)

            loss_mse = (1 / 4) * (torch.norm((fused_image - x1), 2) + torch.norm((fused_image - x2), 2) + torch.norm((fused_image - x3), 2) + torch.norm((fused_image - x4), 2))
            ss_loss = (1 / 4) * (ssim_loss(fused_image, x1) + ssim_loss(fused_image, x2) + ssim_loss(fused_image, x3) + ssim_loss(fused_image, x4))
            [u1, index] = torch.max(images, dim=1)
            u1 = u1.unsqueeze(0)
            light_loss = (1 / (80 * 80 * 64)) * (((fused_image - u1).norm(2)) )
            Loss = (1 / (80 * 80 * 64)) * loss_mse + loss_ncc + loss_smooth + lp*light_loss - sp*ss_loss

            running_loss += Loss.item()

        loss = running_loss / len(val_loader)

        return {'loss': loss}

def train(train_loader,val_loader,scheduler, args, valid_loss_min=999.0,early_stopping_patience=10):
    # Early stopping parameters
    patience_counter = 0

    for e in range(args.epochs):
        train_metrics = train_loop( train_loader,scheduler,e)
        val_metrics = val_loop(val_loader)
        info1 = "Epoch:[{}/{}] train_loss: {:.3f} valid_loss: {:.3f} ".format(e+1,args.epochs,train_metrics["loss"],val_metrics["loss"])

        print(info1)

        with open(args.train_log,'a') as f:
            f.write(info1)

        save_file1 = {"model": unet.state_dict(),
                     "optimizer1": optimizer1.state_dict()}
        save_file2 = {"model": fusenet.state_dict(),
                     "optimizer2": optimizer2.state_dict()}

        # 检查验证损失是否有改善
        if val_metrics['loss'] < valid_loss_min:
            valid_loss_min = val_metrics['loss']
            patience_counter = 0  # 重置耐心计数器
            torch.save(save_file1, args.save_unet_path)
            torch.save(save_file2, args.save_fusenet_path)
        else:
            patience_counter += 1  # 验证损失没有改善，增加耐心计数器

        # 早停条件
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {e + 1} epochs due to no improvement.")
            break
    print("Finished Training!")


def main(args):
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子，以使得结果是确定的

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    patch_size = (80, 80, 64)
    train_dataset = BraTS({'root': args.data_path,  'flist': args.train_txt}, transform=transforms.Compose([CenterCrop(patch_size)]))
    val_dataset = BraTS({'root': args.data_path, 'flist': args.valid_txt}, transform=transforms.Compose([CenterCrop(patch_size)]))
    test_dataset = BraTS({'root': args.data_path, 'flist': args.test_txt}, transform=transforms.Compose([CenterCrop(patch_size)]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=12,  shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,pin_memory=True)

    print("using {} device.".format(device))
    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))
    # img,label = train_dataset[0]

    # 1-坏疽(NT,necrotic tumor core),2-浮肿区域(ED,peritumoral edema),4-增强肿瘤区域(ET,enhancing tumor)
    # 评价指标：ET(label4),TC(label1+label4),WT(label1+label2+label4)

    scheduler = cosine_scheduler(base_value=args.lr, final_value=args.min_lr, epochs=args.epochs,niter_per_ep=len(train_loader), warmup_epochs=args.warmup_epochs, start_warmup_value=5e-4)
    train( train_loader,val_loader,scheduler=scheduler, args=args)

    metrics2 = val_loop(val_loader)
    metrics3 = val_loop(test_loader)

    # 最后再评价一遍所有数据，注意，这里使用的是训练结束的模型参数
    # print("Train -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics1['loss'], metrics1['dice1'],metrics1['dice2'], metrics1['dice3']))
    print("Valid -- loss: {:.3f} ".format(metrics2['loss']))
    print("Test  -- loss: {:.3f} ".format(metrics3['loss']))

if __name__ == '__main__':

    main(args)
