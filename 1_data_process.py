import os
import shutil
import tqdm
import sys
import xml.etree.ElementTree as ET
import json
import random

classes = ['human', 'bicycle', 'motorcycle', 'vehicle']

data_root = os.path.join(sys.path[0], 'data/')

images_path = os.path.join(data_root, 'train', 'Train', 'Week')
labels_path = os.path.join(data_root, 'labels', 'Train', 'Week')

dst_images_path = os.path.join(data_root, "images")
dst_labels_path = os.path.join(data_root, "txts")

xml_path = os.path.join(data_root, "xmls")
test_dir = os.path.join(data_root,'test','Test')
test_images_path = os.path.join(data_root,'test_images')

def process_images_labels():
    for image_folder in os.listdir(os.path.join(images_path, 'frames')):
        day_path = os.path.join(images_path, 'frames', image_folder)
        for splits in os.listdir(day_path):
            for image in os.listdir(os.path.join(day_path, splits)):
                name = str(image_folder) + '_' + splits.split('_')[0] + splits.split('_')[1] + '_' + image
                print(name)
                source_image = os.path.join(images_path, 'frames', image_folder, splits, image)

                dst_image = os.path.join(dst_images_path, name)
                shutil.copy(source_image, dst_image)

    for image_folder in os.listdir(os.path.join(labels_path, 'annotations')):

        day_path = os.path.join(labels_path, 'annotations', image_folder)

        for splits in os.listdir(day_path):
            for image in os.listdir(os.path.join(day_path, splits)):
                name = str(image_folder) + '_' + splits.split('_')[0] + splits.split('_')[1] + '_' + image.replace(
                    'annotations', 'image')
                print(name)
                source_image = os.path.join(labels_path, 'annotations', image_folder, splits, image)

                dst_image = os.path.join(dst_labels_path, name)
                shutil.copy(source_image, dst_image)


def txt_2_xmls():
    label_list = []
    file_list = os.listdir(dst_labels_path)
    for filename in tqdm.tqdm(file_list):
        fin = open(os.path.join(dst_labels_path, filename), 'r')
        image_name = filename.split('.')[0]

        # image_path = image_dir + image_name + ".jpg"
        # img = Image.open(image_dir + image_name + ".jpg")  # 若图像数据是“png”转换成“.png”即可
        # width, height = img.size[0], img.size[1]
        xml_name = os.path.join(xml_path, image_name + '.xml')

        lines_all = fin.readlines()
        lines = []
        number_ship = 0

        for line_all in lines_all:
            line_all = [line_all.split('\n')[0]]
            for line in line_all:
                lines.append(line)

        fin.close()
        with open(xml_name, 'w') as fout:
            fout.write('<annotation>' + '\n')
            fout.write('\t' + '<folder>VOC2007</folder>' + '\n')
            fout.write('\t' + '<filename>' + image_name + '.jpg' + '</filename>' + '\n')

            fout.write('\t' + '<size>' + '\n')
            fout.write('\t\t' + '<width>' + str(384) + '</width>' + '\n')
            fout.write('\t\t' + '<height>' + str(288) + '</height>' + '\n')
            fout.write('\t\t' + '<depth>' + '3' + '</depth>' + '\n')
            fout.write('\t' + '</size>' + '\n')

            fout.write('\t' + '<segmented>' + '0' + '</segmented>' + '\n')

            for line in lines:
                line = line.strip().split(' ')
                bbox = line

                object_id = str(bbox[0])
                class_name = str(bbox[1])

                x1 = eval(bbox[2])
                y1 = eval(bbox[3])
                x2 = eval(bbox[4])
                y2 = eval(bbox[5])

                fout.write('\t' + '<object>' + '\n')
                fout.write('\t\t' + '<name>' + class_name + '</name>' + '\n')
                fout.write('\t\t' + '<pose>' + 'Unspecified' + '</pose>' + '\n')
                fout.write('\t\t' + '<truncated>' + '0' + '</truncated>' + '\n')
                fout.write('\t\t' + '<difficult>' + '0' + '</difficult>' + '\n')
                fout.write('\t\t' + '<id>' + str(object_id) + '</id>' + '\n')
                fout.write('\t\t' + '<bndbox>' + '\n')
                fout.write('\t\t\t' + '<xmin>' + str(x1) + '</xmin>' + '\n')
                fout.write('\t\t\t' + '<ymin>' + str(y1) + '</ymin>' + '\n')
                # pay attention to this point!(0-based)
                fout.write('\t\t\t' + '<xmax>' + str(x2) + '</xmax>' + '\n')
                fout.write('\t\t\t' + '<ymax>' + str(y2) + '</ymax>' + '\n')
                fout.write('\t\t' + '</bndbox>' + '\n')
                fout.write('\t' + '</object>' + '\n')

            fout.write('</annotation>')


def xml_2_json():
    # temp=[]
    images = []
    type = "instances"
    annotations = []
    # assign your categories which contain the classname and calss id
    # the order must be same as the class_nmae
    classes = ['human', 'bicycle', 'motorcycle', 'vehicle']
    categories = [

        {
            "supercategory": 'human',
            "id": 1,
            "name": 'human'
        },
        {
            "supercategory": 'bicycle',
            "id": 2,
            "name": 'bicycle'
        },
        {
            "supercategory": 'motorcycle',
            "id": 3,
            "name": 'motorcycle'
        },
        {
            "supercategory": 'vehicle',
            "id": 4,
            "name": 'vehicle'
        },

    ]
    # load ground-truth from xml annotations
    id_number = 0
    xml_list = os.listdir(xml_path)
    random.shuffle(xml_list)
    select_xml = xml_list[:int(len(xml_list)*0.3)]
    for image_id, label_file_name in tqdm.tqdm(enumerate(select_xml)):

        label_file = os.path.join(xml_path, label_file_name)
        image_file = label_file_name.split('.')[0] + '.jpg'
        tree = ET.parse(label_file)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        images.append({
            "file_name": image_file,
            "height": height,
            "width": width,
            "id": image_id
        })  # id of the image. referenced in the annotation "image_id"

        for anno_id, obj in enumerate(root.iter('object')):
            name = obj.find('name').text
            bbox = obj.find('bndbox')

            cls_id = classes.index(name) + 1
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            xlen = xmax - xmin
            ylen = ymax - ymin
            annotations.append({
                "segmentation": [[xmin, ymax, xmax, ymax, xmax, ymin, xmin, ymin]],
                "area": xlen * ylen,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, xlen, ylen],
                "category_id": cls_id,
                "id": id_number,
                "ignore": 0
            })
            # print([image_file,image_id, cls_id, xmin, ymin, xlen, ylen])
            id_number += 1

    train_json = {"images": images, "annotations": annotations, "categories": categories}
    print(os.path.join(data_root, 'train.json'))
    with open(os.path.join(data_root, 'train.json'), 'w') as json_file:
        json_file.write(json.dumps(train_json, ensure_ascii=False))
        json_file.close()

def process_test_images():

    index = 0
    for month in os.listdir(test_dir):
        for day in tqdm.tqdm(os.listdir(os.path.join(test_dir, month, 'frames'))):
            for splits in os.listdir(os.path.join(test_dir, month, 'frames', day)):
                for image in os.listdir(os.path.join(test_dir, month, 'frames', day, splits)):
                    index += 1
                    name = str(month) + '_' + str(day) + '_' + splits + '_' + image.split('_')[-1]
                    source_image = os.path.join(test_dir, month, "frames", day, splits, image)
                    dst_image = os.path.join(test_images_path, name)
                    shutil.copy(source_image, dst_image)

if __name__ == '__main__':
    process_images_labels()
    txt_2_xmls()
    xml_2_json()
    #process_test_images()
