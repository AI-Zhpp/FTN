import torch
import os
import shutil
from torch.utils.data import DataLoader
from data.dataset_swin_levir import MyTestData
from swin_ynet import Encoder
import torchvision

model = Encoder().cuda()
model.load_state_dict(torch.load('levir_swin.pth'))

test_loader = DataLoader(MyTestData(), shuffle=False, batch_size=1)

outPath = 'levir_swin'
if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.mkdir(outPath)

with torch.no_grad():
    model = model.eval()
    for i, (im1, im2, label_name) in enumerate(test_loader):
        im1 = im1.cuda()
        im2 = im2.cuda()
        label_name = label_name[0]

        outputs = model(im1, im2)
        outputs = outputs[0][0]
        a = outputs[0].unsqueeze(0)

        torchvision.utils.save_image(a, outPath + '/%s' % label_name)
