# Pytorch Code for the CVPR2020 paper: "Perceptual Quality Assessment of Smartphone Photography"
Reference: Y. Fang, H. Zhu, Y. Zeng, K. Ma and Z. Wang, "Perceptual Quality Assessment of Smartphone Photography,"
2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2020, pp. 3674-3683, doi: 10.1109/CVPR42600.2020.00373.

#Note
1. This code is only support to train for the SPAQ dataset, becauase only the SPAQ contains the EXIF tags, and scene catogory labels for training the model.
2. This version only support for training the baseline model, I will release the training code for Multi-Task Learning from EXIF Tags (MT-E), Multi-Task Learning from Image Attributes (MT-A)
and Multi-Task Learning from Scene Semantics (MT-S）when I have free time.
3. Thank you guys very much and I really need Stars to find a job. If my work is helpful to you, please give me some stars ;-).

# Dependencies
- Python 3.6+
- PyTorch 0.4+
- TorchVision

# Training & Testing
Training and testing the model on SPAQ Dataset.
```
python train_test.py
```
# SPAQ Dataset
The SPAQ dataset and the annotations (MOS, image attributes scores, EXIF tags, and scene catogory labels) can be downloaded at the [**Baidu Yun**]
(https://pan.baidu.com/s/18YzAtXb4cGdBGAsxuEVBOw) (Code: b29m) or [**MEGA**](https://mega.nz/folder/SYwUkKjC) (Code: SPaUCc-iWU1VvaZIqmUlnQ).
