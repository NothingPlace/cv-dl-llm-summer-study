import cv2
import numpy as np
import os

def process_images(image_paths, output_dir="output"):
    """
    批量处理图片：转换并保存灰度图、HSV图、缩放图，输出图像信息
    
    参数:
        image_paths: list, 图片路径列表（至少3张）
        output_dir: str, 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, img_path in enumerate(image_paths):
        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"⚠️ 文件不存在: {img_path}")
            continue
        
        # 1. 读取原图
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 无法读取图片: {img_path}")
            continue
        
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 2. 输出原图信息
        height, width, channels = img.shape
        print(f"\n📷 图片 {idx+1}: {os.path.basename(img_path)}")
        print(f"   - 尺寸: {width} x {height}")
        print(f"   - 通道数: {channels} (BGR)")
        print(f"   - 像素值范围: [{img.min()}, {img.max()}]")
        
        # 3. 保存原图（可选，便于对比）
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), img)
        
        # 4. 转换为灰度图并保存
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_path = os.path.join(output_dir, f"{base_name}_gray.jpg")
        cv2.imwrite(gray_path, gray)
        print(f"   ✅ 灰度图已保存: {gray_path}")
        print(f"      - 灰度图尺寸: {gray.shape[1]} x {gray.shape[0]}")
        print(f"      - 灰度图通道数: {1} (单通道)")
        print(f"      - 灰度图像素范围: [{gray.min()}, {gray.max()}]")
        
        # 5. 转换为HSV图并保存
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_path = os.path.join(output_dir, f"{base_name}_hsv.jpg")
        cv2.imwrite(hsv_path, hsv)
        print(f"   ✅ HSV图已保存: {hsv_path}")
        print(f"      - HSV图尺寸: {hsv.shape[1]} x {hsv.shape[0]}")
        print(f"      - HSV图通道数: {hsv.shape[2]}")
        print(f"      - HSV图像素范围: [{hsv.min()}, {hsv.max()}]")
        
        # 6. 缩放图（将原图缩放为原来的一半）
        scaled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        scaled_path = os.path.join(output_dir, f"{base_name}_scaled.jpg")
        cv2.imwrite(scaled_path, scaled)
        print(f"   ✅ 缩放图已保存: {scaled_path}")
        print(f"      - 缩放图尺寸: {scaled.shape[1]} x {scaled.shape[0]}")
        print(f"      - 缩放图通道数: {scaled.shape[2]}")
        print(f"      - 缩放图像素范围: [{scaled.min()}, {scaled.max()}]")
        
        print("-" * 60)


# ============ 使用示例 ============
if __name__ == "__main__":
    # 获取当前 .py 文件所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 1. 准备3张图片路径
    # 构建图片的完整路径
    image_paths = [
        os.path.join(script_dir, "week01-1.jpg"),
        os.path.join(script_dir, "week01-2.jpg"),
        os.path.join(script_dir, "week01-3.jpg"),
    ]
    
    # 2. 如果图片不在当前目录，可以使用绝对路径或相对路径
    # 示例：image_paths = ["C:/Users/xxx/Pictures/photo1.jpg", ...]
    
    # 3. 执行处理
    # 输出目录也设在脚本所在目录下
    output_dir = os.path.join(script_dir, "processed_images")
    process_images(image_paths, output_dir)
    
    print("\n✨ 所有图片处理完成！")
