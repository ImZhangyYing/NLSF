import os
import argparse

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from dataload import *
from Unet import UNet
from fusenet import FuseNet
from Segnet import SegNet
from LOSS import Loss,cal_dice,cosine_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--seed', type=int, default=21)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--min_lr', type=float, default=1e-5)
parser.add_argument('--early_stopping_patience', type=int, default=10)
parser.add_argument('--data_path', type=str, default='./Brats_2021_dataset/BraTS2021_Training_Data')
parser.add_argument('--train_txt', type=str, default='./Brats_2021_dataset/train.txt')
parser.add_argument('--valid_txt', type=str, default='./Brats_2021_dataset/valid.txt')
parser.add_argument('--test_txt', type=str, default='./Brats_2021_dataset/test.txt')
parser.add_argument('--train_log', type=str, default='./model/model.txt')
parser.add_argument('--weights1', type=str, default='./model/unet-model2.pth')
parser.add_argument('--weights2', type=str, default='./model/fuse-model1.pth')
parser.add_argument('--seg_save_path', type=str, default='./model/model.pth')
parser.add_argument('--fuse_save_path', type=str,default='./fuse-model2.pth')
parser.add_argument('--unet_save_path', type=str,default='./unet-model3.pth')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

unet = UNet(1).to(device)
unet_parameters = torch.load(args.weights1)
unet.load_state_dict(unet_parameters['model'])
optimizer1 = optim.Adam(unet.parameters(), lr=0.00001, weight_decay=5e-4)

fusenet = FuseNet().to(device)
fuse_parameters = torch.load(args.weights2)
fusenet.load_state_dict(fuse_parameters['model'])
optimizer2 = optim.Adam(fusenet.parameters(), lr=0.00001, weight_decay=5e-4)

model = SegNet().to(device)
criterion = Loss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

def train_loop(scheduler,train_loader,epoch):
    model.train()
    running_loss = 0
    dice1_train = 0
    dice2_train = 0
    dice3_train = 0
    pbar = tqdm(train_loader)
    for it,(images,masks) in enumerate(pbar):
        # update learning rate according to the schedule
        it = len(train_loader) * epoch + it
        param_group = optimizer.param_groups[0]
        param_group['lr'] = scheduler[it]
        # print(scheduler[it])

        # [b,4,128,128,128] , [b,128,128,128]
        images, masks = images.to(device),masks.to(device)
        # [b,4,128,128,128], 4分割
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
        outputs = model(x1l, x2l, x3l, x4l, fused_image)
        loss = criterion(outputs, masks)
        dice1, dice2, dice3 = cal_dice(outputs,masks)
        pbar.desc = "loss: {:.3f} ".format(loss.item())

        running_loss += loss.item()
        dice1_train += dice1.item()
        dice2_train += dice2.item()
        dice3_train += dice3.item()

        optimizer.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer1.step()
        optimizer2.step()

    loss = running_loss / len(train_loader)

    dice1 = dice1_train / len(train_loader)
    dice2 = dice2_train / len(train_loader)
    dice3 = dice3_train / len(train_loader)
    return {'loss':loss,'dice1':dice1,'dice2':dice2,'dice3':dice3}


def val_loop(val_loader):
    model.eval()
    running_loss = 0
    dice1_val = 0
    dice2_val = 0
    dice3_val = 0
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
            outputs = model(x1l, x2l, x3l, x4l, fused_image)

            loss = criterion(outputs, masks)
            dice1, dice2, dice3 = cal_dice(outputs, masks)

            running_loss += loss.item()
            dice1_val += dice1.item()
            dice2_val += dice2.item()
            dice3_val += dice3.item()
            # pbar.desc = "loss:{:.3f} dice1:{:.3f} dice2:{:.3f} dice3:{:.3f} ".format(loss,dice1,dice2,dice3)

    loss = running_loss / len(val_loader)
    dice1 = dice1_val / len(val_loader)
    dice2 = dice2_val / len(val_loader)
    dice3 = dice3_val / len(val_loader)
    return {'loss':loss,'dice1':dice1,'dice2':dice2,'dice3':dice3}


def train(scheduler,train_loader,val_loader,epochs,train_log,valid_loss_min=999.0,early_stopping_patience=10):
    # Early stopping parameters
    patience_counter = 0

    for e in range(epochs):
        # train for epoch
        train_metrics = train_loop(scheduler,train_loader,e)
        # eval for epoch
        val_metrics = val_loop(val_loader)
        info1 = "Epoch:[{}/{}] train_loss: {:.3f} valid_loss: {:.3f} ".format(e+1,epochs,train_metrics["loss"],val_metrics["loss"])
        info2 = "Train--ET: {:.3f} TC: {:.3f} WT: {:.3f} ".format(train_metrics['dice1'],train_metrics['dice2'],train_metrics['dice3'])
        info3 = "Valid--ET: {:.3f} TC: {:.3f} WT: {:.3f} ".format(val_metrics['dice1'],val_metrics['dice2'],val_metrics['dice3'])
        print(info1)
        print(info2)
        print(info3)
        # Write logs to file
        with open(train_log,'a') as f:
            f.write(info1 + '\n' + info2 + ' ' + info3 + '\n')

        # Save model checkpoints
        save_file1 = {"unet": unet.state_dict(),"optimizer": optimizer1.state_dict()}
        save_file2 = {"fusenet": fusenet.state_dict(),"optimizer": optimizer2.state_dict()}
        save_file3 = {"model": model.state_dict(),"optimizer": optimizer.state_dict()}

        if val_metrics['loss'] < valid_loss_min:
            valid_loss_min = val_metrics['loss']
            patience_counter = 0  # Reset patience counter if validation loss improves
            torch.save(save_file1, args.unet_save_path)
            torch.save(save_file2, args.fuse_save_path)
            torch.save(save_file3, args.seg_save_path)
        else:
            patience_counter += 1  # Increment patience counter if validation loss does not improve
        # Early stopping condition
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


    # data info
    patch_size = (160, 160, 64)
    train_dataset = BraTS({'root': args.data_path, 'flist': args.train_txt},
                          transform=transforms.Compose([CenterCrop(patch_size)]))
    val_dataset = BraTS({'root': args.data_path, 'flist': args.valid_txt},
                        transform=transforms.Compose([CenterCrop(patch_size)]))
    test_dataset = BraTS({'root': args.data_path, 'flist': args.test_txt},
                         transform=transforms.Compose([CenterCrop(patch_size)]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, num_workers=4,  # num_worker=4
                              shuffle=True, pin_memory=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=4, shuffle=False,
                            pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=4, shuffle=False,
                             pin_memory=True)

    print("using {} device.".format(device))
    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))
    # img,label = train_dataset[0]

    # 1-坏疽(NT,necrotic tumor core),2-浮肿区域(ED,peritumoral edema),4-增强肿瘤区域(ET,enhancing tumor)
    # 评价指标：ET(label4),TC(label1+label4),WT(label1+label2+label4)
    scheduler = cosine_scheduler(base_value=args.lr, final_value=args.min_lr, epochs=args.epochs,niter_per_ep=len(train_loader), warmup_epochs=args.warmup_epochs,start_warmup_value=5e-4)


    # # 加载训练模型
    # if os.path.exists(args.seg_save_path):
    #     weight_dict = torch.load(args.seg_save_path, map_location=device)
    #     model.load_state_dict(weight_dict['model'])
    #     optimizer.load_state_dict(weight_dict['optimizer'])
    #     print('Successfully loading checkpoint.')

    train(scheduler,train_loader,val_loader,args.epochs,train_log=args.train_log, early_stopping_patience=args.early_stopping_patience)

    metrics1 = val_loop(train_loader)
    metrics2 = val_loop(val_loader)
    metrics3 = val_loop(test_loader)

    # 最后再评价一遍所有数据，注意，这里使用的是训练结束的模型参数
    print("Train -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics1['loss'], metrics1['dice1'],metrics1['dice2'], metrics1['dice3']))
    print("Valid -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics2['loss'], metrics2['dice1'], metrics2['dice2'], metrics2['dice3']))
    print("Test  -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics3['loss'], metrics3['dice1'], metrics3['dice2'], metrics3['dice3']))


if __name__ == '__main__':

    main(args)
