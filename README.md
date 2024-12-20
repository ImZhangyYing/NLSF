# NLSF
Code of A Nested Self-Supervised Learning Framework for 3-D Semantic Segmentation-Driven Multi-modal Medical Image Fusion

# To download:
Zip File of the train, test, model  
Available at https://pan.baidu.com/s/1scTJ3Q_NtQoAyccY_605HA?pwd=c7kc, Extract code: c7kc

# Dataset Link
You can access the BraTS 2021 dataset through the following channels:

1 Official Source:
The BraTS 2021 dataset has been integrated into the BraTS 2023 dataset.
You need to fill out an application form for BraTS 2023 to gain access.
Link to the application form: https://forms.gle/7cE3Es533usJ4iJq7.
(synapse.org)

2 Kaggle:
You can download parts of the BraTS 2021 dataset, such as Task 1, from Kaggle.
Visit the following link: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1.
(kaggle.com)

3 The Cancer Imaging Archive (TCIA):
TCIA provides the RSNA-ASNR-MICCAI-BraTS-2021 dataset, including brain MRI scans and manual annotations.
You can find the dataset at the following link: https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/.
(cancerimagingarchive.net)

</ul>
<svg></a>Requirements</h1>
<li>Python 3.12.7</li>
<li>PyTorch 2.4.1</li>
</ul>

# To train:
1) For the 1) stage: image reconstruction: run train_unet.py;
2) For the 2) stage: feature fusion: run train_fusenet_gsvl.py;
3) For the 3) stage: segmentation: run train_segnet.py;
# To test:
1) For the 1) stage: run unet_test.py;
2) For the 2) stage: run fusenet_test.py;
3) For the 3) stage: run segnet_test.py;
# If this work is helpful to you, please cite it as:
@article{
  title={A Nested Self-Supervised Learning Framework for 3-D Semantic Segmentation-Driven Multi-modal Medical Image Fusion},
  author={Ying Zhang},
  year={2024},  
}

If you have any question, please email to me (zhangying@mail.ynu.edu.cn).

