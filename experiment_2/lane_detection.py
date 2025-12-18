import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
def read_image(image_path):
    return cv2.imread(image_path)

# 转换为灰度图像
def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊
def gaussian_blur(gray_image, kernel_size=5):
    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)

# Canny边缘检测
def canny_edge_detection(blur_image, low_threshold=50, high_threshold=150):
    return cv2.Canny(blur_image, low_threshold, high_threshold)

# 定义感兴趣区域
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    
    # 创建一个多边形区域,创建模版
    polygon = np.array([[
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]], np.int32)
    
    # 填充多边形区域
    cv2.fillPoly(mask, polygon, 255)
    
    # 只保留感兴趣区域内的边缘
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

# 霍夫变换检测直线
def hough_transform(edges, rho=1, theta=np.pi/180, threshold=30, min_line_len=100, max_line_gap=50):
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# 绘制车道线
def draw_lines(image, lines, color=(0, 255, 0), thickness=10):
    line_image = np.zeros_like(image)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    
    # 将车道线叠加到原始图像上
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return combined_image

# 处理车道线（合并和延伸）
def process_lines(lines, image_height):
    left_lines = []  # 左侧车道线
    right_lines = []  # 右侧车道线
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # 计算斜率
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                
                # 根据斜率区分左右车道线（排除水平线）
                if abs(slope) > 0.5:
                    if slope < 0:
                        left_lines.append((x1, y1, x2, y2, slope))
                    else:
                        right_lines.append((x1, y1, x2, y2, slope))
    
    # 合并左侧车道线
    left_line = merge_lines(left_lines, image_height)
    # 合并右侧车道线
    right_line = merge_lines(right_lines, image_height)
    
    return [left_line, right_line] if left_line and right_line else None

# 合并车道线
def merge_lines(lines, image_height):
    if not lines:
        return None
    
    # 计算所有点的坐标和斜率
    x_coords = []
    y_coords = []
    
    for line in lines:
        x1, y1, x2, y2, slope = line
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    # 使用线性回归拟合车道线
    if len(x_coords) > 1:
        coefficients = np.polyfit(y_coords, x_coords, 1)
        poly = np.poly1d(coefficients)
        
        # 计算车道线的起点和终点
        y1 = image_height  # 底部
        y2 = int(image_height * 0.6)  # 大约图像高度的60%
        
        x1 = int(poly(y1))
        x2 = int(poly(y2))
        
        return [(x1, y1, x2, y2)]
    
    return None

# 主函数
def main(image_path):
    # 1. 读取图像
    image = read_image(image_path)
    if image is None:
        print("无法读取图像，请检查图像路径是否正确")
        return
    
    # 2. 转换为灰度图像
    gray = to_gray(image)
    
    # 3. 高斯模糊
    blur = gaussian_blur(gray)
    
    # 4. Canny边缘检测
    edges = canny_edge_detection(blur)
    
    # 5. 定义感兴趣区域
    masked_edges = region_of_interest(edges)
    
    # 6. 霍夫变换检测直线
    lines = hough_transform(masked_edges)
    
    # 7. 处理车道线
    processed_lines = process_lines(lines, image.shape[0])
    
    # 8. 绘制车道线
    result = draw_lines(image, processed_lines)
    
    # 显示结果
    cv2.imshow("车道线检测结果", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    cv2.imwrite("__street_.jpg", result)
    print("检测结果已保存为lane_detection_result.jpg")

if __name__ == "__main__":
    # 使用示例图像路径（用户可以替换为自己的图像）
    image_path = "street.jpg"
    main(image_path)