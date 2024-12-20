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

from LOSS import gradient_loss_3d,cosine_scheduler
from Unet import UNet
from pytorch_ssim_3D_master import pytorch_ssim


def train_loop(unet,optimizer,scheduler,train_loader, device, epoch):
    unet.train()
    running_loss = 0
    ssim_loss = pytorch_ssim.SSIM3D(window_size=11).to(device)
    pbar = tqdm(train_loader)
    for it, (images, masks) in enumerate(pbar):
        # update learning rate according to the schedule
        it = len(train_loader) * epoch + it
        param_group = optimizer.param_groups[0]
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

        mse_loss = (torch.norm((res_x1 - x1), 2) + torch.norm((res_x2 - x2), 2) + torch.norm((res_x3 - x3), 2) + torch.norm((res_x4 - x4), 2))
        Loss = mse_loss*(1/(80*80*64))

        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        running_loss += Loss.item()
    loss = running_loss / len(train_loader)

    return {'loss': loss}


def val_loop(unet,val_loader,device):
    unet.eval()
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

            mse_loss = (torch.norm((res_x1 - x1), 2) + torch.norm((res_x2 - x2), 2) + torch.norm((res_x3 - x3),2) + torch.norm((res_x4 - x4), 2))
            Loss = mse_loss * (1 / (80 * 80 * 64))
            running_loss += Loss.item()

        loss = running_loss / len(val_loader)

        return {'loss': loss}

def train(unet,optimizer,scheduler,train_loader,val_loader,epochs,device,train_log,save_path=None,valid_loss_min=999.0,early_stopping_patience=10):
    # Early stopping parameters
    patience_counter = 0

    for e in range(epochs):
        train_metrics = train_loop(unet,optimizer,scheduler,train_loader, device, e)
        val_metrics = val_loop(unet,val_loader,device)
        info1 = "Epoch:[{}/{}] train_loss: {:.3f} valid_loss: {:.3f} ".format(e+1,epochs,train_metrics["loss"],val_metrics["loss"])
        print(info1)

        with open(train_log,'a') as f:
            f.write(info1)
        save_file = {"model": unet.state_dict(),
                     "optimizer": optimizer.state_dict()}
        # Check if validation loss improved
        if val_metrics['loss'] < valid_loss_min:
            valid_loss_min = val_metrics['loss']
            patience_counter = 0  # Reset patience counter if validation loss improves
            if save_path is not None:
                torch.save(save_file, save_path)
        else:
            patience_counter += 1  # Increment patience counter if no improvement

        # Early stopping condition
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {e + 1} epochs.")
            break


    print("Finished Training!")


def main(args):
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子，以使得结果是确定的

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patch_size = (80, 80, 64)
    train_dataset = BraTS({'root': args.data_path,  'flist': args.train_txt}, transform=transforms.Compose([CenterCrop(patch_size)]))
    val_dataset = BraTS({'root': args.data_path, 'flist': args.valid_txt}, transform=transforms.Compose([CenterCrop(patch_size)]))
    test_dataset = BraTS({'root': args.data_path, 'flist': args.test_txt}, transform=transforms.Compose([CenterCrop(patch_size)]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=12,shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,pin_memory=True)
    print("using {} device.".format(device))
    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))

    unet = UNet(1).to(device)
    optimizer = optim.Adam(unet.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = cosine_scheduler(base_value=args.lr,final_value=args.min_lr,epochs=args.epochs, niter_per_ep=len(train_loader),warmup_epochs=args.warmup_epochs,start_warmup_value=5e-4)

    train(unet,optimizer,scheduler,train_loader,val_loader,epochs=args.epochs,device=device,train_log=args.train_log,save_path=args.save_path,early_stopping_patience=args.early_stopping_patience)
    metrics1 = val_loop(unet, train_loader, device)
    metrics2 = val_loop(unet, val_loader,device)
    metrics3 = val_loop(unet, test_loader, device)

    # 最后再评价一遍所有数据，注意，这里使用的是训练结束的模型参数
    print("Train  -- loss: {:.3f} ".format(metrics1['loss']))
    print("Valid -- loss: {:.3f} ".format(metrics2['loss']))
    print("Test  -- loss: {:.3f} ".format(metrics3['loss']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--min_lr', type=float, default=0.00001)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='./Brats_2021_dataset/BraTS2021_Training_Data')
    parser.add_argument('--train_txt', type=str, default='./Brats_2021_dataset/train.txt')
    parser.add_argument('--valid_txt', type=str, default='./Brats_2021_dataset/valid.txt')
    parser.add_argument('--test_txt', type=str, default='./Brats_2021_dataset/test.txt')
    parser.add_argument('--train_log', type=str, default='./model/unet-model1.txt')
    parser.add_argument('--save_path', type=str, default='./model/unet-model1.pth')

    args = parser.parse_args()

    main(args)
