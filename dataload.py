import h5py
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import random
from torchvision.transforms import transforms
from torchvision import transforms

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (c, w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[:,w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


#中心裁剪
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (c,w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1] ) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}

#数据类型转换
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}





def process(path):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    # SimpleITK读取图像默认是是 DxHxW，这里转为 HxWxD
    label = sitk.GetArrayFromImage(sitk.ReadImage(path + 'seg.nii.gz')).transpose(1, 2, 0)
    # print(label.shape)
    # 堆叠四种模态的图像，4 x (H,W,D) -> (4,H,W,D)
    # 四种模态的mri图像
    modalities = ('flair', 't1ce', 't1', 't2')
    images = np.stack(
        [sitk.GetArrayFromImage(sitk.ReadImage(path + modal + '.nii.gz')).transpose(1, 2, 0) for modal in modalities],
        0)  # [240,240,155]
    # 数据类型转换
    label[label == 4] = 3
    label = label.astype(np.uint8)
    images = images.astype(np.float32)
    # 对第一个通道求和，如果四个模态都为0，则标记为背景(False)
    mask = images.sum(0) > 0
    mask1 = images.sum(0) == 0
    for k in range(4):
        x = images[k, ...]  #
        y = x[mask]
        # print(x.max(),x.min())
        # print(y.max(), y.min())

      #  # 对背景外的区域进行归一化
        # x[mask] -= y.mean() # 减去非零像素的均值
        # x[mask] /= y.std()  # 除以非零像素的标准差
        x -= y.mean()
        x = (x-x.min())/(x.max()-x.min())
        # print(x[mask].max(),x[mask].min())
        # print(x.max(), x.min())

        images[k, ...] = x
    return images, label

def doit(dset):
    root = dset['root']
    file_list = os.path.join(dset['flist'])
    subjects = open(file_list).read().splitlines()
    # names = ['BraTS2021_' + sub for sub in subjects]
    # paths = [os.path.join(root, name, name + '_') for name in names]
    names = [sub for sub in subjects]
    paths = [os.path.join(root, name, name + '_') for name in names]

    # for path in tqdm(paths):
    #     images, labels = process(path, out_path)

    return paths



class BraTS(Dataset):
    def __init__(self, dset, transform=None):
        self.path = doit(dset)
        self.transform = transform


    def __getitem__(self, idx):
        images, label = process(self.path[idx])

        sample = {'image': images, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        image, label = sample['image'], sample['label']
        # print('label is', np.isnan(label), 'path:',self.paths[item])
        # print('Image is', np.isnan(Image),'path:',self.paths[item])


        # image_patch = image[:, np.newaxis, :, :, :]
        # label_patch = label[np.newaxis, :, :, :]
        image_patch = image[:, :, :, :]
        label_patch = label[:, :, :]
        image_patch = torch.from_numpy(image_patch).float()
        label_patch = torch.from_numpy(label_patch).long()

        return image_patch, label_patch

    def __len__(self):
        return len(self.path)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

if __name__ == '__main__':
    train_set = {
        'root': 'E:/medical6/Ying/Brats_2021_dataset/BraTS2021_Training_Data',  # 四个模态数据所在地址
        'flist': 'E:/medical6/Ying/Brats_2021_dataset/test.txt'}
    test_set = BraTS(train_set, transform=transforms.Compose([CenterCrop((240, 240, 155))]))
    d1 = test_set[0]
    image, label = d1
    Image = image[0, :].squeeze().data.numpy()

    Image = Image.transpose(2, 0, 1)
    savedImg = sitk.GetImageFromArray(Image.astype(np.float32))
    sitk.WriteImage(savedImg, 'E:/medical6/Ying/code/3D/patch/1-image.nii.gz')
    print(image.shape)
    print(label.shape)
    print(np.unique(label))