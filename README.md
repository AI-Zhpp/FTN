# Fully Transformer Network for Change Detection of Remote Sensing Images
****

Paper Links: [Fully Transformer Network for Change Detection of Remote Sensing Images
](https://openaccess.thecvf.com/content/ACCV2022/html/Yan_Fully_Transformer_Network_for_Change_Detection_of_Remote_Sensing_Images_ACCV_2022_paper.html)

by [Tianyu Yan](), [Zifu Wan](), [Pingping Zhang*](https://scholar.google.com/citations?user=MfbIbuEAAAAJ&hl=zh-CN).

## Introduction
****
Recently, change detection (CD) of remote sensing images have achieved great progress with the advances of deep learning. However, current methods generally deliver incomplete CD regions and irregular CD boundaries due to the limited representation ability of the extracted visual features. To relieve these issues, in this work we propose a novel learning framework named Fully Transformer Network (FTN) for remote sensing image CD, which improves the feature extraction from a global view and combines multi-level visual features in a pyramid manner. More specifically, the proposed framework first utilizes the advantages of Transformers in long-range dependency modeling. It can help to learn more discriminative global-level features and obtain complete CD regions. Then, we introduce a pyramid structure to aggregate multi-level visual features from Transformers for feature enhancement. The pyramid structure grafted with a Progressive Attention Module (PAM) can improve the feature representation ability with additional interdependencies through channel attentions. Finally, to better train the framework, we utilize the deeply-supervised learning with multiple boundaryaware loss functions. Extensive experiments demonstrate that our proposed method achieves a new state-of-the-art performance on four public CD benchmarks.

## Update
**** 

* 03/17/2023: The code has been updated.

## Requirements
****
* python 3.5+
* PyTorch 1.1+
* torchvision
* Numpy
* tqdm
* OpenCV

## Preperations
****

For using the codes, please download the public change detection datasets 
(more details are provided in the paper) :
* LEVIR-CD
* WHU-CD
* SYSU-CD
* Google-CD

The processed datasets can be downloaded at this [link](https://drive.google.com/drive/folders/1Knqdxb6g8_7NFKqeHgnp-iUEMRemGCNh?usp=share_link).

Then, run the following codes with your GPUs, and you can get the same results in the above paper.  


## Usage
****

### 1. Download pre-trained Swin Transformer models
* [Get models in this link](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth): SwinB pre-trained on ImageNet22K 


### 2. Prepare data

* Please use *utils/split.py* to split the images to 224*224 first.
* Use *utils/check.py* to check if the labels are binary form. Info will be printed if your label form is incorrect.
* Use *utils/bimap.py* if the labels are not binary.
* You may need to move the aforementioned files to corresponding places.

### 3. Train/Test

- For training, run:

```bash
python train_(name of the dataset).py
```

[//]: # (- If you want to use the SSIM and IOU loss function with CrossEntropy loss funtion together, you just need to remove comments in train.py &#40;below the CrossEntropy loss&#41; and add the loss operation in the loss calculation place.&#41;)
[//]: # (- Especially, when you calculate the IOU loss, you need to convert the images &#40;convert 0->1, 1->0&#41;. Because the image pixels values are mostly 0, and it will influence the IOU loss calculation &#40;Based on IOU loss characteristic&#41;.)

- For prediction, run:
```bash
python test_swin.py 
```

- For evaluation, run:
```bash
python deal_evaluation.py 
```

## Reference
****

* [Swin Transformer](https://github.com/microsoft/Swin-Transformer)

## Contact
****

If you have any problems. Please concat

QQ: 1580329199

Email: tianyuyan2001@gmail.com or wanzifu2000@gmail.com

## Citation
****

If you find our work helpful to your research, please cite with:

```bibtex
@InProceedings{Yan_2022_ACCV,
    author    = {Yan, Tianyu and Wan, Zifu and Zhang, Pingping},
    title     = {Fully Transformer Network for Change Detection of Remote Sensing Images},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2022},
    pages     = {1691-1708}
}
```
