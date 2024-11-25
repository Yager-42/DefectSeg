import json
import os
import os.path as osp
import numpy as np
from utils.dataset import setup_seed
import matplotlib.pyplot as plt
import matplotlib as mpl

setup_seed(20)
str_class = "崩缺，颗粒，脏污，缺金"
classes = {}
for idx, c in enumerate(str_class.split("，")):
    if not c in classes:
        classes[str(c)] = idx


class pic:
    def __init__(self, path):
        self.path = path
        self.classes = [0, 0, 0, 0]


def totalNum(pics):
    dir_path = "data/rawData/1015/train"
    count = os.listdir(dir_path)
    for i in range(0, len(count)):
        path = os.path.join(dir_path, count[i])
        if os.path.isfile(path) and path.endswith("json"):
            data = json.load(open(path))
            pictrue = pic(path)
            for j in range(0, len(data["shapes"])):
                label = data["shapes"][j]["label"]
                if label in classes:
                    pictrue.classes[classes[label]] += 1
            pics.append(pictrue)


def smaller(a, b):
    for i in range(0, len(a)):
        if a[i] < b[i]:
            return 1
    return 0


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def calPrecent(a, b):
    return [x / y for x, y in zip(a, b)]


def splitDataSet():
    pics = []
    totalNum(pics)
    allcounts = [0, 0, 0, 0]
    for pic in pics:
        allcounts = [x + y for x, y in zip(allcounts, pic.classes)]

    # plt.rcParams["font.sans-serif"] = ["SimHei"]
    # mpl.rcParams["axes.unicode_minus"] = False
    # str_class = "缺金，腐蚀，起皮，毛刺，异物污染，表面污染，刮花"
    # xAix = [1, 2, 3, 4, 5, 6, 7]
    # plt.bar(xAix, allcounts, align="center")
    # plt.title("data analyze")
    # plt.xlabel("category")
    # plt.ylabel("count")
    # plt.savefig("data.png")

    traincounts = [x * 0.75 for x in allcounts]
    curCounts = [0, 0, 0, 0]
    trainPics = []
    np.random.shuffle(pics)
    for pic in pics:
        if smaller(curCounts, traincounts):
            curCounts = add(curCounts, pic.classes)
            trainPics.append(pic)
    precent = calPrecent(curCounts, allcounts)
    valPics = [x for x in pics if x not in trainPics]
    trainPics = [x.path for x in trainPics]
    valPics = [x.path for x in valPics]
    return trainPics, valPics


if __name__ == "__main__":
    splitDataSet()
