import torch
import cv2
import numpy as np
import os

class BikeDetector:
    def __init__(self):
        """
        初始化共享单车检测器
        """
        # 加载预训练的YOLOv5模型
        print("正在加载YOLOv5预训练模型...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        print("模型加载完成！")
        
        # 设置置信度阈值
        self.confidence_threshold = 0.5
        
        # COCO数据集中自行车的类别ID为1
        self.bike_class_id = 1
    
    def detect(self, image_path):
        """
        检测图像中的共享单车
        :param image_path: 图像路径
        :return: 检测结果
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 进行检测
        results = self.model(img)
        
        # 解析检测结果
        detections = []
        for *box, conf, cls in results.xyxy[0].numpy():
            if int(cls) == self.bike_class_id and conf >= self.confidence_threshold:
                # 转换为整数坐标
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class': 'bike'
                })
        
        return img, detections
    
    def draw_detections(self, img, detections):
        """
        在图像上绘制检测结果
        :param img: 原始图像
        :param detections: 检测结果
        :return: 绘制了检测结果的图像
        """
        # 创建图像副本
        img_with_detections = img.copy()
        
        # 绘制每个检测结果
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            
            # 绘制边界框
            cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制置信度
            label = f"Bike: {confidence:.2f}"
            cv2.putText(img_with_detections, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_with_detections
    
    def save_result(self, img, output_path):
        """
        保存检测结果图像
        :param img: 绘制了检测结果的图像
        :param output_path: 输出路径
        """
        cv2.imwrite(output_path, img)
        print(f"检测结果已保存到: {output_path}")

def main():
    # 创建检测器实例
    detector = BikeDetector()
    
    # 设置输入和输出目录
    input_dir = 'input_images'
    output_dir = 'output_results'
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        print("正在创建示例图像...")
        os.makedirs(input_dir, exist_ok=True)
        
        # 创建一个简单的示例图像
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (640, 480), (255, 255, 255), -1)
        cv2.putText(img, 'Bike Detection Example', (150, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        example_path = os.path.join(input_dir, 'example.jpg')
        cv2.imwrite(example_path, img)
        print(f"示例图像已创建: {example_path}")
    
    # 获取输入目录中的所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"输入目录中没有找到图像文件: {input_dir}")
        return
    
    # 对每个图像进行检测
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f"detected_{image_file}")
        
        print(f"\n正在检测: {image_file}")
        
        try:
            # 进行检测
            img, detections = detector.detect(image_path)
            
            # 绘制检测结果
            img_with_detections = detector.draw_detections(img, detections)
            
            # 保存结果
            detector.save_result(img_with_detections, output_path)
            
            # 显示检测结果
            cv2.imshow(f"检测结果 - {image_file}", img_with_detections)
            cv2.waitKey(2000)  # 显示2秒
            cv2.destroyAllWindows()
            
            print(f"检测到 {len(detections)} 辆共享单车")
            print(f"结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"检测失败: {e}")

if __name__ == "__main__":
    main()