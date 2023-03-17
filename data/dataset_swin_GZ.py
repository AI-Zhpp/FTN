from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self):
        super(MyData, self).__init__()
        self.train_im_path1 = 'data/CD_Data_GZ/GZ_train_224/A'
        self.train_im_path2 = 'data/CD_Data_GZ/GZ_train_224/B'
        self.train_lb_path = 'data/CD_Data_GZ/GZ_train_224/label'
        self.train_im_num = 3743
        self.train_imgs1 = os.listdir(self.train_im_path1)
        self.train_imgs2 = os.listdir(self.train_im_path2)
        self.train_labels = os.listdir(self.train_lb_path)

    def __len__(self):
        return self.train_im_num

    def __getitem__(self, index):
        img_file1 = os.path.join(self.train_im_path1, self.train_imgs1[index])
        img1 = Image.open(img_file1)
        img_file2 = os.path.join(self.train_im_path2, self.train_imgs2[index])
        img2 = Image.open(img_file2)
        label_file = os.path.join(self.train_lb_path, self.train_labels[index])
        labels = Image.open(label_file)

        im1, im2, lb0, lb1, lb2, lb3 = self.transform(img1, img2, labels)
        lb0 = 1. - lb0[0]
        lb1 = 1. - lb1[0]
        lb2 = 1. - lb2[0]
        lb3 = 1. - lb3[0]

        return im1, im2, lb0, lb1, lb2, lb3  # ,lb4,lb5,lb6,lb7#,edge0,edge1,edge2,edge3

    def transform(self, img1, img2, label):
        transform_img = transforms.Compose([
            transforms.ToTensor(),
        ])
        transform_img_4 = transforms.Compose([transforms.Resize((56, 56), Image.NEAREST),
                                              transforms.ToTensor(),
                                              ])
        transform_img_8 = transforms.Compose([transforms.Resize((28, 28), Image.NEAREST),
                                              transforms.ToTensor(),
                                              ])
        transform_img_16 = transforms.Compose([transforms.Resize((14, 14), Image.NEAREST),
                                               transforms.ToTensor(),
                                               ])

        transform_img_2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        im1 = transform_img_2(img1)
        im2 = transform_img_2(img2)
        label0 = transform_img(label)
        label_4 = transform_img_4(label)
        label_8 = transform_img_8(label)
        label_16 = transform_img_16(label)
        return im1, im2, label0, label_4, label_8, label_16


class MyTestData(Dataset):
    def __init__(self):
        super(MyTestData, self).__init__()
        self.train_im_path1 = 'data/CD_Data_GZ/GZ_test_224/A'
        self.train_im_path2 = 'data/CD_Data_GZ/GZ_test_224/B'
        self.train_lb_path = 'data/CD_Data_GZ/GZ_test_224/label'
        self.train_im_num = 415
        self.train_imgs1 = os.listdir(self.train_im_path1)
        self.train_imgs2 = os.listdir(self.train_im_path2)
        self.train_labels = os.listdir(self.train_lb_path)

    def __len__(self):
        return self.train_im_num

    def __getitem__(self, index):
        img_file1 = os.path.join(self.train_im_path1, self.train_imgs1[index])
        img1 = Image.open(img_file1)
        img_file2 = os.path.join(self.train_im_path2, self.train_imgs2[index])
        img2 = Image.open(img_file2)
        label_file = str(self.train_labels[index][:-4]) + '.png'

        im1, im2 = self.transform(img1, img2)

        return im1, im2, label_file

    def transform(self, img1, img2):
        transform_img_2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        im1 = transform_img_2(img1)
        im2 = transform_img_2(img2)
        return im1, im2
