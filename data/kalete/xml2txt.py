import os
import xml.etree.ElementTree as ET
import random
from os import getcwd

# 注意CenterNet是没有背景类的
classes = ["BoneMan", "Hatter", "FatMan", "LittleMan", "Cowboy", "Werewolf", "Bena",
           "Papjo", "Kokoyi", "GreenDwarf", "Giselle", "Bellett", "StormRider", "Gold", "Door", "Close", ]

# 当前路径
data_path = getcwd()

train_percent = 0.9
Annotations_path = 'Annotations'
total_xml = os.listdir(Annotations_path)
num = len(total_xml)
list = range(num)
num_train = int(num * train_percent)
train_list = random.sample(list, num_train)
ftrain = open('train.txt', 'w')
fval = open('val.txt', 'w')
for i in list:
    xml_path = os.path.join(getcwd(), Annotations_path, total_xml[i])
    xml_content = open(xml_path, 'r')
    write_content = xml_path.replace('Annotations', 'JPGImages').replace('xml', 'jpg')
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        box = (xmlbox.find('xmin').text,
               xmlbox.find('ymin').text,
               xmlbox.find('xmax').text,
               xmlbox.find('ymax').text,)
        write_content += ' ' + str(cls_id) + " " + " ".join([a for a in box])
    if i in train_list:
        ftrain.write(write_content + '\n')
    else:
        fval.write(write_content + '\n')
ftrain.close()
fval.close()
print('共生成', num_train,'张训练图片')
print('共生成', num-num_train,'张验证图片')
