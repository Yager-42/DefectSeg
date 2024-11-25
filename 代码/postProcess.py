from statistics import mean
import cv2
import numpy as np
from PIL import Image
from labelme import utils
from skimage import measure, draw
import math
import os
from main import Model, CmpModel
import torch
from utils.metric import intersectionAndUnion, SegmentationMetric
from utils.sMetric import StreamSegMetrics
import torch.nn.functional as F
import seg_metrics.seg_metrics as sg
import os.path as osp

"""
图像说明：
图像为二值化图像，255白色为目标物，0黑色为背景
要填充白色目标物中的黑色空洞
"""


# https://zhuanlan.zhihu.com/p/63919290
def FillHole(im_in):

    im_floodfill = im_in.copy()
    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint对应像素必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if im_floodfill[i][j] == 0:
                seedPoint = (i, j)
                isbreak = True
                break
        if isbreak:
            break

    # 得到im_floodfill 255填充非孔洞值
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv

    mask = im_out
    im_in = np.where(mask == 255, 7, im_in)

    return im_out
    # 保存结果
    # utils.lblsave(savePath, im_in)


def singleClsPostProcess(img, cls, threadshold, length, width, kernelSize):
    # fetch data
    im_in = np.array(img)
    im_in = np.where(img == cls, 1, 0)

    # fill hole
    im_out = FillHole(im_in.astype(np.uint8))

    k = np.ones((kernelSize, kernelSize), np.uint8)

    # connect close componet
    imOpen = cv2.morphologyEx(im_out.astype(np.uint8), cv2.MORPH_CLOSE, k)
    # delete small componet
    imOpen = cv2.morphologyEx(imOpen.astype(np.uint8), cv2.MORPH_OPEN, k)

    # delete componet smaller than threadshold
    connect = measure.label(imOpen, connectivity=1)
    props = measure.regionprops(connect)
    areaOfCon = []
    wOfCon = []
    lOfCon = []

    for ia in range(len(props)):
        areaOfCon.append(props[ia].area)
        bBox = props[ia].bbox
        len1 = bBox[2] - bBox[0]
        len2 = bBox[3] - bBox[1]

        lOfCon.append(math.sqrt(math.pow(len1, 2) + math.pow(len2, 2)))
        wOfCon.append(min(len1, len2))
    idxNeedDel = []
    for i in range(len(areaOfCon)):
        area = areaOfCon[i]
        area = int(area)
        l = int(lOfCon[i])
        w = int(wOfCon[i])
        if cls in [1, 2, 3, 5]:
            if (
                area > threadshold * 0.8625 * 0.8625
                or l > length * 0.8625
                or w > width * 0.8625
            ):
                continue
            idxNeedDel.append(props[i].label)
        else:
            if (
                area > threadshold * 0.8625 * 0.8625
                and l > length * 0.8625
                and w > width * 0.8625
            ):
                continue
            idxNeedDel.append(props[i].label)

    for i in idxNeedDel:
        connect = np.where(connect == i, 0, connect)

    mask = np.where(connect != 0, 1, 0)

    im = img
    im = np.where(im == cls, 0, im)
    im_out = np.where(mask == 1, cls, im)

    # save img
    # utils.lblsave(savePath, im_out)
    return im_out


def multiClsPostProcess(im_in, threadsholds, kernel, length, width):
    # fetch data
    im_in = np.array(im_in)
    classes = np.unique(im_in)
    for cls in classes:
        if cls == 0:
            continue
        lenLimit = length[cls - 1]
        widLimit = width[cls - 1]
        tmp = lenLimit
        lenLimit = max(tmp, widLimit)
        widLimit = min(tmp, widLimit)
        imgProceessed = singleClsPostProcess(
            im_in, cls, threadsholds[cls - 1], lenLimit, widLimit, kernel[cls - 1]
        )
        im_in = imgProceessed
    # utils.lblsave(savePath, im_in)
    return im_in


def isNG(imgPre, threadsholds, kernel, length, width):
    img = multiClsPostProcess(imgPre, threadsholds, kernel, length, width)
    classes = np.unique(img)
    if len(classes) > 1:
        return True, img
    return False, img


fMetric = True
fPostProcess = False
fPFNet = True
fSavePredict = False


def evalLostNG(imgs_dir, masks_dir):
    labels = [1, 2, 3, 4, 5, 6, 7]
    top_img_dir = imgs_dir
    img_names = []
    img_dirs = os.listdir(top_img_dir)
    for dir in img_dirs:
        for name in os.listdir(os.path.join(top_img_dir, dir)):
            img_names.append(os.path.join(top_img_dir, dir, name))

    if fPFNet:
        model0 = Model.load_from_checkpoint("check/mixpf/epoch=99-val_miou=0.6628.ckpt")
        model1 = Model.load_from_checkpoint(
            "check/mixpf/epoch=155-val_miou=0.6538.ckpt"
        )
        model2 = Model.load_from_checkpoint(
            "check/mixpf/epoch=174-val_miou=0.6549.ckpt"
        )
    else:
        model0 = CmpModel.load_from_checkpoint(
            "check/0101/epoch=86-val_miou=0.9318.ckpt"
        )
        model1 = CmpModel.load_from_checkpoint(
            "check/0101/epoch=87-val_miou=0.9318.ckpt"
        )
        model2 = CmpModel.load_from_checkpoint(
            "check/0101/epoch=88-val_miou=0.9373.ckpt"
        )

    model0.eval()
    model1.eval()
    model2.eval()

    metric = SegmentationMetric(9, isTest=True, isArgMax=True)
    sMetric = StreamSegMetrics(9)

    threadsholds = [999999999, 999999999, 999999999, 500, 1000, 500, 1000]
    kernel = [5, 0, 5, 5, 0, 5, 5]
    length = [60, 50, 30, 20, 30, 0, 50]
    width = [60, 50, 30, 20, 30, 0, 10]

    totalNG = 0
    NGCnt = 0
    totalUnion = 0
    totalInter = 0

    diceAll = [0, 0, 0, 0, 0, 0, 0, 0]
    iouAll = [0, 0, 0, 0, 0, 0, 0, 0]
    precisionAll = [0, 0, 0, 0, 0, 0, 0, 0]
    recallAll = [0, 0, 0, 0, 0, 0, 0, 0]
    cnt = [0, 0, 0, 0, 0, 0, 0, 0]

    tpAll = [0, 0, 0, 0, 0, 0, 0, 0]
    fpAll = [0, 0, 0, 0, 0, 0, 0, 0]
    fnAll = [0, 0, 0, 0, 0, 0, 0, 0]

    wrongImg = []
    for imgName in img_names:
        img = cv2.imread(imgName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        img = np.array(img) / 255
        img = torch.from_numpy(
            np.expand_dims(img.transpose(2, 0, 1), 0).astype(np.float32)
        ).to(torch.device("cuda:0"))

        with torch.no_grad():
            predict0 = model0(img)
            predict1 = model1(img)
            predict2 = model2(img)
            predict = predict0 + predict1 + predict2

        predict = torch.argmax(predict, 1).unsqueeze(1).to(dtype=torch.float32)
        # predict = F.interpolate(predict, (256, 256), mode="nearest")
        predict = np.array(predict.to(dtype=torch.long).cpu()).squeeze(0).squeeze(0)
        predict_ = predict

        if fMetric:
            dir_name = imgName.split(".")[0].split("/")[-2]
            img_name = imgName.split(".")[0].split("/")[-1]
            mask_path = os.path.join(masks_dir, dir_name, img_name + ".png")

            mask = np.array(Image.open(mask_path))
            # mask = np.where(mask != 0, mask - 1, 0)
            mask = np.array(
                cv2.resize(
                    mask,
                    (512, 512),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
            metricsAll = sg.write_metrics(
                labels=[1, 2, 3, 4, 5, 6, 7, 8],  # exclude background if needed
                gdth_img=mask,
                pred_img=predict_,
                metrics=["dice", "jaccard", "precision", "recall"],
                TPTNFPFN=True,
            )
            tp, tn, fp, fn = (
                metricsAll[0]["TP"],
                metricsAll[0]["TN"],
                metricsAll[0]["FP"],
                metricsAll[0]["FN"],
            )

            tpAll = [i + j for i, j in zip(tpAll, tp)]
            fpAll = [i + j for i, j in zip(fpAll, fp)]
            fnAll = [i + j for i, j in zip(fnAll, fn)]

            diceAll = [i + j for i, j in zip(diceAll, metricsAll[0]["dice"])]
            iouAll = [i + j for i, j in zip(iouAll, metricsAll[0]["jaccard"])]
            precisionAll = [
                i + j for i, j in zip(precisionAll, metricsAll[0]["precision"])
            ]
            recallAll = [i + j for i, j in zip(recallAll, metricsAll[0]["recall"])]
            for i in range(len(cnt)):
                if metricsAll[0]["dice"][i] != 0.0:
                    cnt[i] += 1

            metric.update(
                torch.from_numpy(predict_).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
            )

            sMetric.update(mask, predict_)
            if fSavePredict:
                utils.lblsave(
                    osp.join("evalRes/mask", imgName.split("/")[-1] + ".png"), mask
                )

        if fPostProcess:
            NG, predict_ = isNG(predict, threadsholds, kernel, length, width)
            if NG:
                NGCnt += 1
            else:
                print(imgName)
                wrongImg.append(imgName)
                a = np.unique(predict)
                b = np.unique(predict_)
                utils.lblsave("savePre.png", predict)
                utils.lblsave("savePostPre.png", predict_)
            totalNG += 1
        else:
            if len(np.unique(predict)) > 1:
                NGCnt += 1
            totalNG += 1

        if fSavePredict:
            utils.lblsave(
                osp.join("evalRes/my", imgName.split("/")[-1] + ".png"), predict_
            )

    if fPostProcess:
        threadsholds = [999999999, 999999999, 999999999, 500, 1000, 500, 1000]
        kernel = [5, 0, 5, 5, 0, 5, 5]
        length = [60, 50, 30, 20, 30, 0, 50]
        width = [60, 50, 30, 20, 30, 0, 10]
        for imgName in wrongImg:
            img = cv2.imread(imgName)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            img = np.array(img) / 255
            img = torch.from_numpy(
                np.expand_dims(img.transpose(2, 0, 1), 0).astype(np.float32)
            ).to(torch.device("cuda:0"))

            with torch.no_grad():
                predict0 = model0(img)
                predict1 = model1(img)
                predict2 = model2(img)
                predict = predict0 + predict1 + predict2

            predict = torch.argmax(predict, 1).unsqueeze(1).to(dtype=torch.float32)
            # predict = F.interpolate(predict, (256, 256), mode="nearest")
            predict = np.array(predict.to(dtype=torch.long).cpu()).squeeze(0).squeeze(0)
            predict_ = predict

            NG, predict_ = isNG(predict, threadsholds, kernel, length, width)
            if not NG:
                NGCnt -= 1
            else:
                print(imgName)
                a = np.unique(predict)
                b = np.unique(predict_)
                utils.lblsave("savePre.png", predict)
                utils.lblsave("savePostPre.png", predict_)

    if fPostProcess:
        print(1.0 - NGCnt / totalNG)
        print(NGCnt)

    if fMetric:
        print(metric.get())
        print(sMetric.get_results())

        diceAll = [j / i for i, j in zip(cnt, diceAll)]
        iouAll = [j / i if i != 0 else 0 for i, j in zip(cnt, iouAll) if i != 0]
        precisionAll = [j / i for i, j in zip(cnt, precisionAll)]
        recallAll = [j / i for i, j in zip(cnt, recallAll)]

        print([diceAll, mean(diceAll)])
        print([iouAll, mean(iouAll)])
        print([precisionAll, mean(precisionAll)])
        print([recallAll, mean(recallAll)])

        dictMetric = {}

        smooth = 0.0000001
        tmpDice = [
            i * 2 / (j + k + 2 * i + smooth) for i, j, k in zip(tpAll, fpAll, fnAll)
        ]

        tmpIOU = [i / (i + j + k + smooth) for i, j, k in zip(tpAll, fpAll, fnAll)]

        tmpPrecision = [i / (i + j + smooth) for i, j in zip(tpAll, fpAll)]

        tmpRecall = [i / (i + j + smooth) for i, j in zip(tpAll, fnAll)]

        f1 = [2 * i * j / (i + j + smooth) for i, j in zip(tmpPrecision, tmpRecall)]

        dictMetric["dice"] = tmpDice
        dictMetric["IOU"] = tmpIOU
        dictMetric["Precision"] = tmpPrecision
        dictMetric["Recall"] = tmpRecall
        dictMetric["f1"] = f1
        dictMetric["MDice"] = sum(tmpDice) / len(tmpDice)
        dictMetric["MIOU"] = sum(tmpIOU) / len(tmpIOU)
        dictMetric["MPrecision"] = sum(tmpPrecision) / len(tmpPrecision)
        dictMetric["MRecall"] = sum(tmpRecall) / len(tmpRecall)
        dictMetric["MF1"] = sum(f1) / len(f1)
        print(dictMetric)


# data/反面val/Images
# data/反面ok
evalLostNG("data/mix(1)/Images", "data/mix(1)/Mask")
