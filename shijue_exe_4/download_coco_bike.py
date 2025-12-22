import os
import json
import requests
import zipfile
import shutil
import cv2
import numpy as np
from tqdm import tqdm

class COCOBikeDownloader:
    def __init__(self, output_dir='dataset'):
        """
        初始化COCO自行车数据集下载器
        :param output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.coco_years = ['2017']  # 可以添加其他年份
        self.coco_base_url = 'http://images.cocodataset.org'
        self.coco_annotations_url = 'http://images.cocodataset.org/annotations'
        
        # COCO数据集中自行车的类别ID
        self.bike_category_id = 1
    
    def download_file(self, url, save_path):
        """
        下载文件并显示进度条
        :param url: 文件URL
        :param save_path: 保存路径
        """
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        
        print(f"正在下载: {os.path.basename(save_path)}")
        with open(save_path, 'wb') as file, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
    
    def extract_zip(self, zip_path, extract_to):
        """
        解压ZIP文件
        :param zip_path: ZIP文件路径
        :param extract_to: 解压目录
        """
        print(f"正在解压: {os.path.basename(zip_path)}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    
    def get_bike_annotations(self, annotations_file):
        """
        获取自行车类别的标注
        :param annotations_file: 标注文件路径
        :return: 自行车标注列表
        """
        print(f"正在解析标注文件: {os.path.basename(annotations_file)}")
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        # 获取所有图像ID
        image_ids = set()
        bike_annotations = []
        
        for ann in annotations['annotations']:
            if ann['category_id'] == self.bike_category_id:
                image_ids.add(ann['image_id'])
                bike_annotations.append(ann)
        
        # 获取图像信息
        images = [img for img in annotations['images'] if img['id'] in image_ids]
        
        return images, bike_annotations
    
    def convert_to_yolo(self, bbox, img_width, img_height):
        """
        将COCO格式的边界框转换为YOLO格式
        :param bbox: COCO格式的边界框 [x, y, width, height]
        :param img_width: 图像宽度
        :param img_height: 图像高度
        :return: YOLO格式的边界框 [x_center, y_center, width, height]
        """
        x, y, w, h = bbox
        
        # 计算YOLO格式的坐标（归一化）
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        
        return [x_center, y_center, width, height]
    
    def create_yolo_labels(self, images, annotations, split):
        """
        创建YOLO格式的标签文件
        :param images: 图像列表
        :param annotations: 标注列表
        :param split: 数据集分割 (train/val)
        """
        # 创建目录
        img_dir = os.path.join(self.output_dir, split, 'images')
        label_dir = os.path.join(self.output_dir, split, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        
        # 为每个图像创建标签文件
        print(f"正在创建{split}集的YOLO标签...")
        
        # 创建图像ID到图像信息的映射
        img_id_to_info = {img['id']: img for img in images}
        
        # 按图像ID分组标注
        anns_by_img = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in anns_by_img:
                anns_by_img[img_id] = []
            anns_by_img[img_id].append(ann)
        
        # 处理每个图像
        for img_id, img_info in img_id_to_info.items():
            img_name = img_info['file_name']
            img_path = os.path.join('coco', split + '2017', img_name)
            
            # 检查图像文件是否存在
            if not os.path.exists(img_path):
                print(f"图像不存在: {img_path}")
                continue
            
            # 复制图像到输出目录
            shutil.copy(img_path, os.path.join(img_dir, img_name))
            
            # 创建标签文件
            label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
            
            with open(label_path, 'w') as f:
                if img_id in anns_by_img:
                    for ann in anns_by_img[img_id]:
                        # 转换边界框格式
                        yolo_bbox = self.convert_to_yolo(ann['bbox'], img_info['width'], img_info['height'])
                        # YOLO格式: <class_id> <x_center> <y_center> <width> <height>
                        f.write(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
    
    def download_coco(self):
        """
        下载COCO数据集
        """
        if not os.path.exists('coco'):
            os.makedirs('coco')
        
        for year in self.coco_years:
            for split in ['train', 'val']:
                # 下载图像
                img_zip_url = f"{self.coco_base_url}/zips/{split}{year}.zip"
                img_zip_path = os.path.join('coco', f"{split}{year}.zip")
                
                if not os.path.exists(img_zip_path):
                    self.download_file(img_zip_url, img_zip_path)
                    self.extract_zip(img_zip_path, 'coco')
                else:
                    print(f"图像文件已存在: {img_zip_path}")
            
            # 下载标注
            ann_zip_url = f"{self.coco_annotations_url}/annotations_trainval{year}.zip"
            ann_zip_path = os.path.join('coco', f"annotations_trainval{year}.zip")
            
            if not os.path.exists(ann_zip_path):
                self.download_file(ann_zip_url, ann_zip_path)
                self.extract_zip(ann_zip_path, 'coco')
            else:
                print(f"标注文件已存在: {ann_zip_path}")
    
    def download_and_prepare(self, max_images=None):
        """
        下载并准备COCO自行车数据集
        :param max_images: 最大图像数量（可选）
        """
        # 下载COCO数据集
        self.download_coco()
        
        # 为每个分割创建YOLO数据集
        for year in self.coco_years:
            for split in ['train', 'val']:
                # 获取标注文件路径
                ann_file = os.path.join('coco', 'annotations', f"instances_{split}{year}.json")
                
                # 获取自行车标注
                images, annotations = self.get_bike_annotations(ann_file)
                
                # 如果指定了最大图像数量，只使用部分图像
                if max_images is not None and max_images > 0:
                    images = images[:max_images]
                    # 更新标注，只保留选定图像的标注
                    selected_img_ids = {img['id'] for img in images}
                    annotations = [ann for ann in annotations if ann['image_id'] in selected_img_ids]
                
                print(f"在{split}{year}集中找到 {len(images)} 张自行车图像，{len(annotations)} 个标注")
                
                # 创建YOLO标签
                self.create_yolo_labels(images, annotations, split)
        
        # 创建数据配置文件
        self.create_data_config()
        
        print("\n数据集准备完成！")
        print(f"数据集保存在: {self.output_dir}")
        print(f"训练集图像数量: {len(os.listdir(os.path.join(self.output_dir, 'train', 'images')))}")
        print(f"验证集图像数量: {len(os.listdir(os.path.join(self.output_dir, 'val', 'images')))}")
    
    def create_data_config(self):
        """
        创建YOLOv5所需的数据配置文件
        """
        data_config = {
            'path': os.path.abspath(self.output_dir),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 1,
            'names': ['bike']
        }
        
        config_path = os.path.join(self.output_dir, 'data.yaml')
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"数据配置文件已创建: {config_path}")
    
    def cleanup(self):
        """
        清理临时文件
        """
        if os.path.exists('coco'):
            print("正在清理COCO原始数据...")
            shutil.rmtree('coco')
            print("清理完成！")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='下载COCO数据集的自行车部分')
    parser.add_argument('--output-dir', type=str, default='dataset', help='输出目录')
    parser.add_argument('--max-images', type=int, default=None, help='最大图像数量')
    parser.add_argument('--cleanup', action='store_true', help='清理原始COCO数据')
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = COCOBikeDownloader(output_dir=args.output_dir)
    
    # 下载并准备数据集
    downloader.download_and_prepare(max_images=args.max_images)
    
    # 清理临时文件
    if args.cleanup:
        downloader.cleanup()
    
    print("\n使用方法:")
    print(f"python train_bike_detector.py --data-dir {args.output_dir} --epochs 100 --batch-size 16")

if __name__ == '__main__':
    main()