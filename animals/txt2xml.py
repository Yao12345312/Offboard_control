# coding:utf-8

import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

def convert_yolo_to_voc(txt_folder, output_folder, classes, image_folder):
    """
    将 YOLO 格式的 TXT 文件转换为 Pascal VOC 格式的 XML 文件。

    :param txt_folder: 存放 YOLO 格式 TXT 文件的文件夹路径
    :param output_folder: 存放转换后 XML 文件的文件夹路径
    :param classes: 类别列表
    :param image_folder: 存放对应图片的文件夹路径（用于获取图像宽高）
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历 TXT 文件夹中的所有文件
    for txt_file in os.listdir(txt_folder):
        if not txt_file.endswith('.txt'):
            continue

        # 获取对应的图片文件名
        image_name = os.path.splitext(txt_file)[0] + '.jpg'
        image_path = os.path.join(image_folder, image_name)

        # 假设图片宽高（如果需要从图片中读取宽高，可以使用 PIL 或 OpenCV）
        # 这里需要根据实际情况修改
        width, height = 720, 1160  # 替换为实际图片的宽高

        # 创建 XML 根节点
        annotation = Element('annotation')

        # 添加文件名和路径
        SubElement(annotation, 'folder').text = os.path.basename(image_folder)
        SubElement(annotation, 'filename').text = image_name
        SubElement(annotation, 'path').text = image_path

        # 添加图像大小信息
        size = SubElement(annotation, 'size')
        SubElement(size, 'width').text = str(width)
        SubElement(size, 'height').text = str(height)
        SubElement(size, 'depth').text = '3'  # 假设为 RGB 图像

        # 添加 segmented 标签
        SubElement(annotation, 'segmented').text = '0'

        # 读取 YOLO 格式的 TXT 文件
        txt_path = os.path.join(txt_folder, txt_file)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            # YOLO 格式：class_id x_center y_center width height
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # 转换为 Pascal VOC 格式的边界框坐标
            xmin = int((x_center - box_width / 2) * width)
            ymin = int((y_center - box_height / 2) * height)
            xmax = int((x_center + box_width / 2) * width)
            ymax = int((y_center + box_height / 2) * height)

            # 创建 object 节点
            obj = SubElement(annotation, 'object')
            SubElement(obj, 'name').text = classes[class_id]
            SubElement(obj, 'pose').text = 'Unspecified'
            SubElement(obj, 'truncated').text = '0'
            SubElement(obj, 'difficult').text = '0'

            # 添加边界框信息
            bndbox = SubElement(obj, 'bndbox')
            SubElement(bndbox, 'xmin').text = str(xmin)
            SubElement(bndbox, 'ymin').text = str(ymin)
            SubElement(bndbox, 'xmax').text = str(xmax)
            SubElement(bndbox, 'ymax').text = str(ymax)

        # 将 XML 写入文件
        xml_str = tostring(annotation)
        dom = parseString(xml_str)
        xml_pretty = dom.toprettyxml(indent="  ")

        xml_file = os.path.join(output_folder, os.path.splitext(txt_file)[0] + '.xml')
        with open(xml_file, 'w') as f:
            f.write(xml_pretty)


# 示例用法
if __name__ == "__main__":
    # YOLO 格式 TXT 文件夹路径
    txt_folder = r"C:\Users\Administrator\Desktop\project\electronic_design\yolov5\animals\labels"
    # 输出 XML 文件夹路径
    output_folder = r"C:\Users\Administrator\Desktop\project\electronic_design\yolov5\animals\Annotations"
    # 类别列表（按顺序定义）
    classes = ["plate"]
    # 对应图片文件夹路径
    image_folder = r"C:\Users\Administrator\Desktop\project\electronic_design\yolov5\animals\images"

    convert_yolo_to_voc(txt_folder, output_folder, classes, image_folder)