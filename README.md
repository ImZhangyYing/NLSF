# NLSF
Code of A Nested Self-Supervised Learning Framework for 3-D Semantic Segmentation-Driven Multi-modal Medical Image Fusion
# To download:
File of the train, test, model and datasets 
Available at https://pan.baidu.com/s/1scTJ3Q_NtQoAyccY_605HA?pwd=c7kc, Extract code: c7kc
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

