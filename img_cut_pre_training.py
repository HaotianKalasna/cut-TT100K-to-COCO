import os 
import cv2
import json
import shutil

def anno2id(anno):
    c1, c2 = anno[0:2]
    if c1 == 'i':
        if c2 == 'l':
            id = 3
        else:
            id = 0
    elif c1 == 'p':
        if c2 in ['h','m','w','a']:
            id = 6
        elif c2 in ['e','g','s']:
            id = 5
        elif c2 == 'l':
            id = 4
        elif c2 == 'r':
            id = 7
        else:
            id = 1
    else:
        id = 2
    return id

cut_height = 64
cut_width = 64
folders = ["train","test"]
folders_2 = range(8)

# 目录设置和创建
data_path = "data/"
data_out_path = "data_classification/"
if os.path.exists(data_out_path):
    shutil.rmtree(data_out_path)
for folder in folders:
    for folder_2 in folders_2:
        os.makedirs(data_out_path+folder+"/"+"%d/"%folder_2, 0o777)

with open(data_path+"annotations.json") as f:
    json_data = json.load(f)


# 导入原标注文件中的信息
imgs = json_data["imgs"]
id = dict(train=0, test=0)
for index in imgs :
    # if id["trainval"]>100:
    #     break
    img_path = imgs[index]["path"]
    folder = img_path.split("/")[0]
    if folder == "other":
        continue
    img = cv2.imread(data_path+img_path)
    if img is None:
        continue
    objects = imgs[index]["objects"]
    # 跳过没有目标的图片
    if len(objects) == 0:
        continue
    for obj in objects:
        cut_img = img[int(obj["bbox"]["ymin"]):int(obj["bbox"]["ymax"]),int(obj["bbox"]["xmin"]):int(obj["bbox"]["xmax"])]
        h, w = cut_img.shape[0:2]
        if h == 0 or w == 0:
            continue
        BLACK = [0, 0, 0]
        if h < w:
            cut_img = cv2.resize(cut_img, (64, int(64*h/w)))
            b = (64-int(64*h/w))/2
            cut_img = cv2.copyMakeBorder(cut_img, int(b), int(b), 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
        else :
            cut_img = cv2.resize(cut_img, (int(64*w/h), 64))
            a = (64-int(64*w/h))/2
            cut_img = cv2.copyMakeBorder(cut_img, 0, 0, int(a), int(a), cv2.BORDER_CONSTANT, value=BLACK)
        
        cut_img = cv2.resize(cut_img, (64, 64), interpolation=cv2.INTER_AREA)

        cat = anno2id(obj["category"])
        cut_img_path = data_out_path + folder + "/%d/"%cat + "%06d.png"%id[folder]
        cv2.imwrite(cut_img_path, cut_img)
        print("img: %d exported"%(id["test"]+id["train"]))
        id[folder] = id[folder] + 1