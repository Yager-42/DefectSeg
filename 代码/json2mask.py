import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils

"""
制作自己的语义分割数据集需要注意以下几点：
1、我使用的labelme版本是3.16.7，建议使用该版本的labelme，有些版本的labelme会发生错误，
   具体错误为：Too many dimensions: 3 > 2
   安装方式为命令行pip install labelme==3.16.7
2、此处生成的标签图是8位彩色图，与视频中看起来的数据集格式不太一样。
   虽然看起来是彩图，但事实上只有8位，此时每个像素点的值就是这个像素点所属的种类。
   所以其实和视频中VOC数据集的格式一样。因此这样制作出来的数据集是可以正常使用的。也是正常的。
"""


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


if __name__ == "__main__":
    dir_path = "data/rawData/1015/val"
    jpgs_path = "data/1015/Images/val/" + dir_path.split("/")[-1]
    pngs_path = "data/1015/Mask/val/" + dir_path.split("/")[-1]
    check_mkdir(jpgs_path)
    check_mkdir(pngs_path)

    str_class = "崩缺，颗粒，脏污，缺金"

    classes = {"_background_": 0}
    for idx, c in enumerate(str_class.split("，")):
        if not c in classes:
            classes[str(c)] = idx + 1

    count = os.listdir(dir_path)
    for i in range(0, len(count)):
        path = os.path.join(dir_path, count[i])

        if os.path.isfile(path) and path.endswith("json"):
            data = json.load(open(path))
            if len(data["shapes"]) == 0:
                continue

            if data["imageData"]:
                imageData = data["imageData"]
            else:
                imagePath = os.path.join(os.path.dirname(path), data["imagePath"])
                with open(imagePath, "rb") as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode("utf-8")

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = classes

            data["shapes"] = list(
                filter(lambda shape: len(shape["points"]) > 2, data["shapes"])
            )

            # for idx,shape in enumerate(data['shapes']):
            #    if data['shapes'][idx]['label'].startswith("颗粒"):
            #        data['shapes'][idx]['label'] = "颗粒"
            #        continue
            #    if data['shapes'][idx]['label']=="崩缺-白":
            #        data['shapes'][idx]['label'] = "崩缺白"
            #        continue
            #    if data['shapes'][idx]['label']=="崩瓷-白":
            #        data['shapes'][idx]['label'] = "崩瓷白"
            #        continue
            #    data['shapes'][idx]['label'] = data['shapes'][idx]['label'].split('-')[0]

            data["shapes"] = [
                shape for shape in data["shapes"] if shape["label"] in classes
            ]

            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)

            lbl = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)

            new = np.zeros([np.shape(img)[0], np.shape(img)[1]])
            for name in label_names:
                # index_json = label_names.index(name)
                index_json = label_name_to_value[name]
                # index_all = classes.index(name)
                index_all = label_name_to_value[name]
                new = new + index_all * (np.array(lbl) == index_json)

            m = np.unique(new)
            if len(m) == 1 and m[0] == 0:
                continue

            PIL.Image.fromarray(img).save(
                osp.join(jpgs_path, count[i].split(".")[0] + ".jpg")
            )
            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0] + ".png"), new)
            # PIL.Image.fromarray(new).convert('L').save(osp.join(pngs_path, count[i].split(".")[0]+'.png'))
            print(
                "Saved "
                + count[i].split(".")[0]
                + ".jpg and "
                + count[i].split(".")[0]
                + ".png"
            )
            print("class: ", m)
