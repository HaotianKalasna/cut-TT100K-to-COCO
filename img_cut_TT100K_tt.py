import os
import shutil 
import cv2
import json

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

cut_height = 128
cut_width = 128
folders = ["trainval","test"]

# 目录设置和创建
data_path = "data/"
coco_path = "cocott/"
if os.path.exists(coco_path):
    shutil.rmtree(coco_path)
for folder in folders:
    os.makedirs(coco_path+folder+"/", 0o777)
os.makedirs(coco_path+"annotations/", 0o777)

with open(data_path+"annotations.json") as f:
    json_data = json.load(f)

# 准备用于生成三个json文件的dict
trainval_json_data = dict(images=[],annotations=[],categories=[])
test_json_data  = dict(images=[],annotations=[],categories=[])
categories = [dict(id=0, name="indication"),
              dict(id=1, name="prohibition"),
              dict(id=2, name="warnning"),
              dict(id=3, name="minspeedlimit"),
              dict(id=4, name="maxspeedlimit"),
              dict(id=5, name="giveway"),
              dict(id=6, name="hmwalimit"),
              dict(id=7, name="speedlimitremove"),
              ]
trainval_json_data["categories"]   = categories
test_json_data["categories"]    = categories

coco_images         = dict(trainval=[],test=[])
coco_annotations    = dict(trainval=[],test=[])

# 导入原标注文件中的信息
imgs = json_data["imgs"]

id = dict(trainval=0,test=0,trainvalanno=0,testanno=0)
for index in imgs :
    # if id["trainval"]>100:
    #     break
    img_path = imgs[index]["path"]
    folder = img_path.split("/")[0]
    if folder == "other":
        continue
    img = cv2.imread(data_path+img_path)
    img_width, img_height = img.shape[0:2]
    objects = imgs[index]["objects"]
    # 跳过没有目标的图片
    if len(objects) == 0:
        continue
    for x in range(0, img_width-cut_width, cut_width-32):
        for y in range (0, img_height-cut_height, cut_height-32):
            # 找出切分后图中的目标
            cut_objs = []
            for obj in objects:
                exp =  ((int(obj["bbox"]["xmin"]) > x)\
                    and (int(obj["bbox"]["ymin"]) > y)\
                    and (int(obj["bbox"]["xmax"]) < (x + cut_width))\
                    and (int(obj["bbox"]["ymax"]) < (y + cut_height)))
                if exp:
                    obj["bbox"]["xmin"] = float(int(obj["bbox"]["xmin"]) - x)
                    obj["bbox"]["ymin"] = float(int(obj["bbox"]["ymin"]) - y)
                    obj["bbox"]["xmax"] = float(int(obj["bbox"]["xmax"]) - x)
                    obj["bbox"]["ymax"] = float(int(obj["bbox"]["ymax"]) - y)
                    cut_objs.append(obj)

            # 跳过没有目标的切分
            if cut_objs == []:
                continue

            # 将train中一部分分出来做val
            if folder == "train":
                folder = "trainval"

            # 添加切分图片信息至images
            image = dict()
            image["id"]         = id[folder]
            image["file_name"]  = "%06d.png"%(id[folder])
            image["width"]      = cut_width
            image["height"]     = cut_height
            coco_images[folder].append(image)

            # 添加切分标注信息至annotations
            for obj in cut_objs:
                annotation = dict()
                annotation["id"]            = id[folder+"anno"]
                annotation["image_id"]      = image["id"]
                annotation["category_id"]   = anno2id(obj["category"])
                annotation["iscrowd"]       = 0
                annotation["bbox"]          = [obj["bbox"]["xmin"],obj["bbox"]["ymin"],obj["bbox"]["xmax"]-obj["bbox"]["xmin"],obj["bbox"]["ymax"]-obj["bbox"]["ymin"]]
                annotation["area"]          = annotation["bbox"][2] * annotation["bbox"][3]
                annotation["segmentation"]  = [[obj["bbox"]["xmin"],obj["bbox"]["ymin"],obj["bbox"]["xmax"],obj["bbox"]["ymin"],obj["bbox"]["xmax"],obj["bbox"]["ymax"],obj["bbox"]["xmin"],obj["bbox"]["ymax"]]]
                coco_annotations[folder].append(annotation)
                id[folder+"anno"] = id[folder+"anno"] + 1 

            # 将图片存储到相应文件夹
            cut_img_path = coco_path + folder + "/%06d.png"%(id[folder])
            cut_img = img[y:y+cut_height, x:x+cut_width]
            cv2.imwrite(cut_img_path, cut_img)
            print("img: %06d saved"%(id["trainval"]+id["test"]))
            id[folder] = id[folder] + 1
            
trainval_json_data["images"]       = coco_images["trainval"]
trainval_json_data["annotations"]  = coco_annotations["trainval"]
trainval_json = json.dumps(trainval_json_data)
with open(coco_path+"annotations/trainval.json","w") as f:
    f.write(trainval_json)
test_json_data["images"]        = coco_images["test"]
test_json_data["annotations"]   = coco_annotations["test"]
test_json = json.dumps(test_json_data)
with open(coco_path+"annotations/test.json","w") as f:
    f.write(test_json)




