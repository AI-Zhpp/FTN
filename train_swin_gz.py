import torch
import torch.nn as nn
import tqdm
from my_scheduler import LR_Scheduler
from swin_ynet import Encoder
from data.dataset_swin_GZ import MyData
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
model = Encoder().cuda()

import pytorch_iou
import pytorch_ssim

deal = nn.Softmax(dim=1)


def all_loss(pred, gt):
    ce_loss = nn.CrossEntropyLoss()
    ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True).cuda()
    iou_loss = pytorch_iou.IOU().cuda()
    ce_out = ce_loss(pred, gt.long())
    ssim_out = 1 - ssim_loss(deal(pred), gt)
    iou_out = iou_loss(deal(pred), gt)
    loss = ce_out + ssim_out + iou_out
    return loss


model = model.train()
ce_loss = nn.CrossEntropyLoss()
ssim_loss = pytorch_ssim.SSIM(window_size=7, size_average=True).cuda()
iou_loss = pytorch_iou.IOU().cuda()
LR = 0.01
LR_VGG = 0.00001
EPOCH = 80
scheduler = LR_Scheduler('cos', LR, EPOCH, 3743 // 10 + 1)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005, nesterov=False)


def make_optimizer(LR, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if "encoder1" in key:
            lr = LR * 0.1
        else:
            lr = LR
        params += [{"params": [value], "lr": lr}]
    optimizer = getattr(torch.optim, "SGD")(params, momentum=0.9, weight_decay=0.0005, nesterov=False)
    return optimizer


train_loader = DataLoader(MyData(),
                          shuffle=True,
                          batch_size=10,
                          pin_memory=True,
                          num_workers=16,
                          )

losses0 = 0
losses1 = 0
losses2 = 0
losses3 = 0
losses4 = 0
losses5 = 0
losses6 = 0
losses7 = 0
losses8 = 0
losses9 = 0
losses10 = 0
losses11 = 0

print(len(train_loader))


def adjust_learning_rate(optimizer, epoch, start_lr):
    if epoch % 20 == 0:  # epoch != 0 and
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.1
        print(param_group["lr"])


loss_least = 100000
for epoch_num in range(EPOCH):
    print(epoch_num)
    adjust_learning_rate(optimizer, epoch_num, LR)
    print('LR is:', optimizer.state_dict()['param_groups'][0]['lr'])
    show_dict = {'epoch': epoch_num}

    loss_all = 0
    for i_batch, (im1, im2, label0, label1, label2, label3) in enumerate(
            tqdm.tqdm(train_loader, ncols=60, postfix=show_dict)):  # ,edge0,edge1,edge2,edge3
        im1 = im1.cuda()
        im2 = im2.cuda()
        label0 = label0.cuda()
        label1 = label1.cuda()
        label2 = label2.cuda()
        label3 = label3.cuda()

        outputs = model(im1, im2)

        loss0 = ce_loss(outputs[0], label0.long())
        loss1 = ce_loss(outputs[1], label1.long())
        loss2 = ce_loss(outputs[2], label2.long())
        loss3 = ce_loss(outputs[3], label3.long())

        loss4 = 1. - ssim_loss(deal(outputs[0]), label0)
        loss5 = 1. - ssim_loss(deal(outputs[1]), label1)
        loss6 = 1. - ssim_loss(deal(outputs[2]), label2)
        loss7 = 1. - ssim_loss(deal(outputs[3]), label3)

        loss8 = iou_loss(deal(outputs[0]), label0)
        loss9 = iou_loss(deal(outputs[1]), label1)
        loss10 = iou_loss(deal(outputs[2]), label2)
        loss11 = iou_loss(deal(outputs[3]), label3)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11
        loss_all += loss

        losses0 += loss0
        losses1 += loss1
        losses2 += loss2
        losses3 += loss3
        losses4 += loss4
        losses5 += loss5
        losses6 += loss6
        losses7 += loss7
        losses8 += loss8
        losses9 += loss9
        losses10 += loss10
        losses11 += loss11

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i_batch % 100 == 0:
            print(i_batch, '|', 'losses0: {:.3f}'.format(losses0.data), '|', 'losses1: {:.3f}'.format(losses1.data),
                  '|', 'losses2: {:.3f}'.format(losses2.data), '|', 'losses3: {:.3f}'.format(losses3.data), '|',
                  'losses4: {:.3f}'.format(losses4.data), '|', 'losses5: {:.3f}'.format(losses5.data), '|',
                  'losses6: {:.3f}'.format(losses6.data), '|', 'losses7: {:.3f}'.format(losses7.data),
                  'losses8: {:.3f}'.format(losses8.data), '|', 'losses9: {:.3f}'.format(losses9.data), '|',
                  'losses10: {:.3f}'.format(losses10.data), '|', 'losses11: {:.3f}'.format(losses11.data))

            losses0 = 0
            losses1 = 0
            losses2 = 0
            losses3 = 0
            losses4 = 0
            losses5 = 0
            losses6 = 0
            losses7 = 0
            losses8 = 0
            losses9 = 0
            losses10 = 0
            losses11 = 0

    if loss_all <= loss_least:
        loss_least = loss_all
        torch.save(model.state_dict(), 'new_try3.pth')
        print('\n', 'epoch:', epoch_num, 'epoch loss:', loss_all)
