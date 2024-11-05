import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(xml_folder, output_folder, classes):
    os.makedirs(output_folder, exist_ok=True)
    
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith(".xml"):
            continue
        
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()
        
        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)
        
        yolo_format_lines = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in classes:
                continue
            class_id = classes.index(class_name)
            
            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)
            
            x_center = (x_min + x_max) / 2.0 / img_width
            y_center = (y_min + y_max) / 2.0 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            yolo_format_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
        output_txt_file = os.path.join(output_folder, xml_file.replace(".xml", ".txt"))
        with open(output_txt_file, "w") as f:
            f.write("\n".join(yolo_format_lines))

# Укажите ваши классы объектов (например, солнечные пятна и солнечные вспышки)
classes = ["sunspots", "solar flares"]

# Конвертация аннотаций для обучающего набора
convert_voc_to_yolo("data/annotations/train/", "data/images/labeled/train/", classes)

# Конвертация аннотаций для валидационного набора
convert_voc_to_yolo("data/annotations/val/", "data/images/labeled/val/", classes)
