#########################################################

This Repo. is used for our ACCV2022 paper: 

Fully Transformer Network for Change Detection of Remote Sensing Images

(https://arxiv.org/abs/2210.00757)

https://openaccess.thecvf.com/content/ACCV2022/papers/Yan_Fully_Transformer_Network_for_Change_Detection_of_Remote_Sensing_Images_ACCV_2022_paper.pdf

#########################################################

For using the codes, please download the public change detection datasets (more details are provided in the paper): 

LEVIR-CD,

WHU-CD,

SYSU-CD,

Google-CD.

Besides, please install the Pytorch as the Official Suggestions.

Then, run the following codes with your GPUs. You can get the same results in the above paper.  

The codes can be download at 

Link：https://pan.baidu.com/s/1ellGHVTXNRp0gmeMEPJYcA

Extracting code：b99t

######################## Usage ###########################

### 1. Download pre-trained Swin Transformer models
* [Get models in this link](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth): SwinB pre-trained on ImageNet22K 


### 2. Prepare data

Please use split.py to split the images first, and use bimap.py, deal.py and check.py to make the GT images become binary images.

### 3. Environment

Please prepare an environment with python=3.7, opencv, torch and torchvision. And then when running the program, it reminds you to install whatever library you need.

### 4. Train/Test

- If you want to try the model, you can directly use the train.py.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

- If you want to use the SSIM and IOU loss function with CrossEntropy loss funtion together, you just need to remove comments in train.py (below the CrossEntropy loss) and add the loss operation in the loss calculation place.
- Especially, when you calculate the IOU loss, you need to convert the images (convert 0->1, 1->0). Because the image pixels values are mostly 0, and it will influence the IOU loss calculation (Based on IOU loss characteristic).

- If you want to obtain the result, you can directly use the test.py.
```bash
python test.py 
```

## Reference
* [Swin Transformer](https://github.com/microsoft/Swin-Transformer)

########################################################

If you have any problems. Please concat

tianyuyan2001@gmail.com or wanzifu2000@gmail.com or zhpp@dlut.edu.cn

########################################################

If you find our work can help your reseach, please consider citing:

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
