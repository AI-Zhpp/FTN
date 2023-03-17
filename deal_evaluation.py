import cv2
import numpy as np
import os
import shutil

refile = 'levir_swin'  # file obtained by test_swin.py
outPath = 'levir_swin_deal'  # temp file
if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.mkdir(outPath)
name = os.listdir(refile)

for i in range(len(name)):
    label_file = os.path.join(refile, name[i])
    a = cv2.imread(label_file, 0)

    b = 2. * np.mean(a)
    b, photo = cv2.threshold(a, 20, 255, cv2.THRESH_BINARY)
    cv2.imwrite(outPath + '/' + name[i], photo)

Files_path = outPath
labels_num = len(os.listdir(Files_path))
print(labels_num)

outPath = 'try'  # temp file2
if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.mkdir(outPath)

for i in range(labels_num):
    image_dir = os.path.join(Files_path, str(os.listdir(Files_path)[i]))
    image_path = os.path.join(image_dir)
    img = cv2.imread(image_path)
    img = img[:, :, 0]
    lb0 = cv2.merge([img * 255.])
    cv2.imwrite(outPath + '/' + str(os.listdir(Files_path)[i]), lb0)

gt_path = 'xxx/label'  # corresponding gt path
pred_path = outPath
labels_num = len(os.listdir(gt_path))

print(labels_num)


def P_R_IoU(gt, pred):
    predict_precision = 0
    predict_recall = 0
    tp = 0
    tn = 0
    for k in range(len(gt)):

        img1 = gt[k]
        img2 = pred[k]
        for i in range(224):
            for j in range(224):
                if not (int(img1[i, j]) - int(img2[i, j])) and img2[i, j] == 255:
                    tp += 1  # TP value
                if img1[i, j] == img2[i, j] and img2[i, j] == 0:
                    tn += 1  # TN value

        predict_precision += np.sum(np.reshape(img2, (img2.size,))) / 255
        predict_recall += np.sum(np.reshape(img1, (img1.size,))) / 255
        print(k)
    predict_iou = predict_precision + predict_recall - tp
    return tp / predict_precision, tp / predict_recall, tp / predict_iou, (tn + tp) / (predict_iou + tn)


def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


def get_average(list_):
    sum_ = 0
    for _ in list_:
        sum_ += _
    return sum_ / len(list_)


gt_list = []
pred_list = []

for i in range(labels_num):
    gt = os.path.join(gt_path, str(os.listdir(gt_path)[i]))
    pred = os.path.join(pred_path, str(os.listdir(pred_path)[i]))
    gt_path1 = os.path.join(gt)
    pred_path1 = os.path.join(pred)
    gt1 = cv2.imread(gt_path1, flags=0)
    pred1 = cv2.imread(pred_path1, flags=0)
    gt_list.append(gt1)
    pred_list.append(pred1)

precision_res, recall_res, iou_res, OA = P_R_IoU(gt_list, pred_list)

f1_score = f1_score(precision_res, recall_res)
print('precision:', precision_res, 'recall:', recall_res, '\n', 'f1:', f1_score, 'IoU:', iou_res, 'OA:', OA)
