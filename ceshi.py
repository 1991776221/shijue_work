import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

# 1. 图像读取（使用自己拍摄的图像）
def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像，请检查路径是否正确")
    # 转换为RGB格式（方便后续直方图显示）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

# 2. 灰度转换
def to_gray(image):
    # 手动实现灰度转换（不调用cv2.cvtColor）
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return gray.astype(np.uint8)

# 3. 手动实现卷积（不调用任何滤波函数包）
def convolve(image, kernel):
    img_height, img_width = image.shape
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2

    # 图像填充
    padded_img = np.zeros((img_height + 2 * pad_size, img_width + 2 * pad_size), dtype=np.float32)
    padded_img[pad_size:pad_size + img_height, pad_size:pad_size + img_width] = image.astype(np.float32)

    # 使用as_strided提取窗口（非函数包调用，属于基础数组操作）
    stride = padded_img.strides
    window_shape = (img_height, img_width, kernel_size, kernel_size)
    window_strides = (stride[0], stride[1], stride[0], stride[1])
    
    windows = as_strided(
        padded_img,
        shape=window_shape,
        strides=window_strides,
        writeable=False
    )
    
    # 计算卷积
    output = np.sum(windows * kernel, axis=(2, 3))
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

# 4. 手动实现颜色直方图（不调用函数包）
def compute_histogram(image):
    # 对于RGB图像，分别计算三个通道的直方图
    channels = ['r', 'g', 'b']
    histograms = {}
    
    for i, channel in enumerate(channels):
        # 初始化直方图数组（0-255共256个 bins）
        hist = np.zeros(256, dtype=np.int32)
        # 遍历图像像素，统计每个像素值的出现次数
        for row in image[:, :, i]:
            for pixel in row:
                hist[pixel] += 1
        histograms[channel] = hist
    
    return histograms

# 5. 手动实现纹理特征提取（基于灰度共生矩阵，不调用函数包）
def extract_texture_features(gray_image):
    # 灰度共生矩阵计算（简化版，计算0度方向，距离1）
    gray_level = 256
    glcm = np.zeros((gray_level, gray_level), dtype=np.int32)
    
    # 遍历图像，统计相邻像素对
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1] - 1):  # 避免越界
            current = gray_image[i, j]
            neighbor = gray_image[i, j + 1]  # 右侧相邻像素
            glcm[current, neighbor] += 1
    
    # 从共生矩阵中提取特征
    # 1. 能量
    energy = np.sum(glcm ** 2) / np.sum(glcm) if np.sum(glcm) > 0 else 0
    # 2. 熵
    glcm_normalized = glcm / np.sum(glcm) if np.sum(glcm) > 0 else 0
    entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + 1e-10))  # 加小值避免log(0)
    # 3. 对比度
    contrast = 0
    for i in range(gray_level):
        for j in range(gray_level):
            contrast += (i - j) ** 2 * glcm_normalized[i, j]
    
    return {
        'energy': energy,
        'entropy': entropy,
        'contrast': contrast,
        'glcm': glcm  # 保存完整共生矩阵
    }

# 6. 结果可视化与保存
def visualize_and_save_results(original, sobel_x, sobel_y, custom_filter, histograms, texture_features, save_path='results/'):
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 保存原始图像和滤波结果
    cv2.imwrite(save_path + 'original.jpg', cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_path + 'sobel_x.jpg', sobel_x)
    cv2.imwrite(save_path + 'sobel_y.jpg', sobel_y)
    cv2.imwrite(save_path + 'custom_filter.jpg', custom_filter)
    
    # 绘制并保存直方图
    plt.figure(figsize=(12, 4))
    for i, (channel, hist) in enumerate(histograms.items()):
        plt.subplot(1, 3, i + 1)
        plt.bar(range(256), hist, color=channel)
        plt.title(f'{channel.upper()} Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path + 'histograms.png')
    plt.close()
    
    # 保存纹理特征
    np.save(save_path + 'texture_features.npy', texture_features)
    
    print(f"所有结果已保存至 {save_path} 目录")

# 主函数
def main(image_path='cat.jpg'):  # 使用目录中已存在的cat.jpg图像
    # 步骤1：读取图像
    img_bgr, img_rgb = read_image(image_path)
    print(f"图像读取完成，尺寸: {img_bgr.shape}")
    
    # 步骤2：转换为灰度图
    gray = to_gray(img_bgr)
    print("灰度图转换完成")
    
    # 步骤3：定义卷积核
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    custom_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)  # 给定卷积核
    
    # 步骤4：应用滤波
    sobel_x = convolve(gray, sobel_x_kernel)
    sobel_y = convolve(gray, sobel_y_kernel)
    custom_filtered = convolve(gray, custom_kernel)
    print("所有滤波操作完成")
    
    # 步骤5：计算颜色直方图
    histograms = compute_histogram(img_rgb)
    print("颜色直方图计算完成")
    
    # 步骤6：提取纹理特征
    texture_features = extract_texture_features(gray)
    print("纹理特征提取完成")
    
    # 步骤7：可视化并保存结果
    visualize_and_save_results(img_rgb, sobel_x, sobel_y, custom_filtered, histograms, texture_features)
    
    # 显示结果（可选）
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.imshow('Original', img_bgr)
    cv2.namedWindow('Sobel X', cv2.WINDOW_NORMAL)
    cv2.imshow('Sobel X', sobel_x)
    cv2.namedWindow('Sobel Y', cv2.WINDOW_NORMAL)
    cv2.imshow('Sobel Y', sobel_y)
    cv2.namedWindow('Custom Filter', cv2.WINDOW_NORMAL)
    cv2.imshow('Custom Filter', custom_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main('cat.jpg')  # 直接使用cat.jpg图像运行程序
