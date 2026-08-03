import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process_images(image_paths,output_dir):

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

            #读取及灰度化
            
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            median = cv2.medianBlur(gray,3)

            equalize = cv2.equalizeHist(median)

            # sobel = cv2.Sobel(median,cv2.CV_8U,1,0,ksize=3)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = cv2.convertScaleAbs(sobel)

            ret,binary = cv2.threshold(sobel,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)

            # element1 = cv2.getStructuringElement(cv2.MORPH_RECT,(30,9))
            # element2 = cv2.getStructuringElement(cv2.MORPH_RECT,(24,6))
            element1 = cv2.getStructuringElement(cv2.MORPH_RECT,(45,18))
            element2 = cv2.getStructuringElement(cv2.MORPH_RECT,(36,9))

            dilation = cv2.dilate(binary,element2,iterations=1)
            erosion = cv2.erode(dilation,element1,iterations=1)

            region = []
            contours,hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            for i in range (len(contours)):
                cnt = contours[i]

                area = cv2.contourArea(cnt)

                rect = cv2.minAreaRect(cnt)

                box = cv2.boxPoints(rect)
                box = np.int32(box)

                height = abs(box[0][1]-box[2][1])
                width = abs(box[0][0]-box[2][0])
                x, y, w, h = cv2.boundingRect(cnt)
                rect_area = w * h
                solidity = area / rect_area
                if(height > width * 1.5):
                    continue
                # if solidity > 0.8:   
                #     continue

                region.append(box)

            processed_img=img.copy()

            for box in region:
                 print(box)
                 cv2.drawContours(processed_img,[box],0,(0,255,0),2)

            origin_path = os.path.join(output_dir, f"{base_name}_origin.jpg")
            cv2.imwrite(origin_path, img)
            median_path = os.path.join(output_dir, f"{base_name}_median.jpg")
            cv2.imwrite(median_path, median)
            sobel_path = os.path.join(output_dir, f"{base_name}_sobel.jpg")
            cv2.imwrite(sobel_path, sobel)
            processed_path = os.path.join(output_dir, f"{base_name}_processed.jpg")
            cv2.imwrite(processed_path, processed_img)
            print(f"   ✅ 文字区域已提取: {processed_path}")

            fig, axes = plt.subplots(2, 4, figsize=(10, 10))  # 2x2 网格

            #在每个子区域显示图片
            axes[0, 0].imshow(gray, cmap='gray')
            axes[0, 0].set_title('gray')
            axes[0, 1].imshow(median, cmap='gray')
            axes[0, 1].set_title('median')
            axes[0, 2].imshow(equalize, cmap='gray')
            axes[0, 2].set_title('equalize')
            axes[0, 3].imshow(sobel, cmap='gray')
            axes[0, 3].set_title('sobel')
            axes[1, 0].imshow(binary, cmap='gray')
            axes[1, 0].set_title('binary')
            axes[1, 1].imshow(dilation, cmap='gray')
            axes[1, 1].set_title('dilation')
            axes[1, 2].imshow(erosion, cmap='gray')
            axes[1, 2].set_title('erosion')
            axes[1, 3].imshow(processed_img)
            axes[1, 3].set_title('processed_img')

            # 关闭坐标轴
            for ax in axes.flat:
                ax.axis('off')

            plt.tight_layout()
            plt.show()


# ============ 使用示例 ============
if __name__ == "__main__":
    # 获取当前 .py 文件所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建图片的完整路径
    image_paths = [
        os.path.join(script_dir, "week02-1.jpg"),
        os.path.join(script_dir, "week02-2.jpg"),
        os.path.join(script_dir, "week02-3.png"),
    ]
    
    # 输出目录也设在脚本所在目录下
    output_dir = os.path.join(script_dir, "processed_images")
    process_images(image_paths, output_dir)
    
    print("\n✨ 所有图片处理完成！")






