# opencv基础

## 一、图像的基本操作

### 1. 图像读取 - cv2.imread()

```python
cv2.imread(filename, flags=cv2.IMREAD_COLOR)
#示例
img_color = cv2.imread('image.jpg', cv2.IMREAD_COLOR)    # 彩色读取
img_gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE) # 灰度读取
```

`filename`：字符串类型，图像文件的路径（支持绝对路径和相对路径）
`flags`：整数类型，指定读取图像的方式

1. cv2.IMREAD_COLOR (1)：默认值，加载彩色图像，忽略透明度（3通道BGR）
2. cv2.IMREAD_GRAYSCALE (0)：以灰度模式加载图像（单通道）
3. cv2.IMREAD_UNCHANGED (-1)：加载图像包含 Alpha 通道（4通道BGRA）
4. cv2.IMREAD_ANYDEPTH (2)：保留图像的原始深度（16位/32位），否则转为8位
5. cv2.IMREAD_ANYCOLOR (4)：以原始颜色格式读取

返回值： numpy.ndarray 类型的图像矩阵，读取失败返回 None

#### 读取及修改像素点

```python
point = img[88,142]#img[行,列]
img[88,142] = [255, 255, 255]
#分别获取BGR通道像素(opencv读取图片为BGR格式)
blue = img[88,142,0]
green = img[88,142,1]
red = img[88,142,2]
#该区域设置为白色
img[100:200, 150:250] = [255,255,255]
#Numpy形式
blue = img.item(78, 100, 0)
green = img.item(78, 100, 1)
red = img.item(78, 100, 2)
img.itemset((78, 100, 0), 100)
img.itemset((78, 100, 1), 100)
img.itemset((78, 100, 2), 100)
```

| 颜色空间 | 全称 | 通道组成 | 主要用途 |
| --------- | ------ | --------- |--------- |
| **BGR** | Blue-Green-Red | 蓝(B)、绿(G)、红(R) | OpenCV默认格式，由历史原因导致与常见RBG不同 |
| **RGB** | Red-Green-Blue | 红(R)、绿(G)、蓝(B) | 显示器、网页、PIL/Pillow |
| **HSV** | Hue-Saturation-Value | 色调(H)、饱和度(S)、明度(V) | 颜色识别、图像分割 |

### 2. 图像显示 - cv2.imshow()

```python
cv2.imshow(winname, mat)
```

`winname`：字符串类型，显示窗口的名称

- 相同的窗口名会复用窗口

- 可以使用中文（但建议使用英文以避免编码问题）

`mat`：numpy.ndarray 类型，要显示的图像矩阵

```python
cv2.waitKey(delay)：等待按键,用于避免图片只显示一瞬间
```

`delay`：等待时间（毫秒），0 表示无限等待，按任意键继续

返回值：按键的 ASCII 码

```python
if key == 27:  # ESC键
    cv2.destroyAllWindows()
```

可以通过以上代码实现按esc退出

```python
cv2.destroyAllWindows()：关闭所有窗口
cv2.destroyWindow(winname)：关闭指定窗口
```

### 3. 图像保存 - cv2.imwrite()

```python
cv2.imwrite(filename, img, params=None)
# 示例 - 保存为高质量JPEG
cv2.imwrite('output.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])
# 示例 - 保存为压缩PNG
cv2.imwrite('output.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
```

`filename`：字符串，保存的文件路径（扩展名决定格式）

`img`：numpy.ndarray，要保存的图像

`params`：列表类型，编码参数（可选）

- [cv2.IMWRITE_JPEG_QUALITY, 95]：JPEG质量（0-100），默认95

- [cv2.IMWRITE_PNG_COMPRESSION, 3]：PNG压缩级别（0-9），默认3

- [cv2.IMWRITE_WEBP_QUALITY, 80]：WebP质量（1-100）

返回值： 布尔类型，True 表示保存成功

## 二、图像基本属性与转换

### 1. 图像属性

```python
# 图像形状
height, width, channels = img.shape  # 彩色图像
height, width = gray_img.shape       # 灰度图像

# 图像大小（总像素数）
total_pixels = img.size

# 图像数据类型
dtype = img.dtype  # 通常为 uint8
```

`shape`：元组，(高度, 宽度, 通道数)，灰度图只有 (高度, 宽度)

`size`：图像总像素数（height × width × channels）

`dtype`：数据类型，通常为 uint8 (0-255)

### 2. 颜色空间转换 - cv2.cvtColor()

```python
cv2.cvtColor(src, code, dst=None, dstCn=None)
# 示例
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

`src`：输入图像（numpy.ndarray）

`code`：颜色空间转换代码

- `cv2.COLOR_BGR2GRAY`：BGR转灰度

- `cv2.COLOR_BGR2HSV`：BGR转HSV（色调H:0-179, 饱和度S:0-255, 明度V:0-255）

- `cv2.COLOR_BGR2RGB`：BGR转RGB

- `cv2.COLOR_BGR2LAB`：BGR转LAB

- `cv2.COLOR_HSV2BGR`：HSV转BGR

- `cv2.COLOR_GRAY2BGR`：灰度转BGR

`dst`：输出图像（可选）

`dstCn`：输出图像的通道数（可选，0表示自动）

| 颜色空间 | 全称 | 通道组成 | 主要用途 |
| --------- | ------ | --------- |--------- |
| **BGR** | Blue-Green-Red | 蓝(B)、绿(G)、红(R) | OpenCV默认格式，由历史原因导致与常见RBG不同 |
| **RGB** | Red-Green-Blue | 红(R)、绿(G)、蓝(B) | 显示器、网页、PIL/Pillow |
| **HSV** | Hue-Saturation-Value | 色调(H)、饱和度(S)、明度(V) | 颜色识别、图像分割 |

#### 手动转换

```python
#拆分通道
b, g, r = cv2.split(img)
#显示拆分后图像
cv2.imshow("B", b)
cv2.imshow("G", g)
cv2.imshow("R", r)
#拆分后的图像以灰度图显示
#拆分通道
b = cv2.split(img)[0]
g = np.zeros((rows,cols),dtype=img.dtype)
r = np.zeros((rows,cols),dtype=img.dtype)
#合并通道
m = cv2.merge([b, g, r])
cv2.imshow("Merge", m)#显示blue图
#灰度化（加权法）
gray_np = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8) #.astype(np.uint8)用于将浮点数运算的结果转换回uint8格式
```

## 三、图像几何变换

### 1. 图像缩放 - cv2.resize()

```python
cv2.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)

# 按目标尺寸缩放
resized = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)

# 按比例缩放
resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
```

`src`：输入图像

`dsize`：目标图像大小，元组 (width, height)

`dst` 是 **destination（目标）** 的缩写，表示**输出图像**。它是一个可选参数，用于指定缩放后的图像存储位置，不影响返回值

如果为 (0,0)，则通过 fx 和 fy 计算

fx：水平缩放因子（默认0） fy：垂直缩放因子（默认0）

`interpolation`：插值方法

- `cv2.INTER_NEAREST`：最近邻插值（速度快，质量低）

- `cv2.INTER_LINEAR`：双线性插值（默认，适合放大）

- `cv2.INTER_CUBIC`：双三次插值（速度慢，质量好，适合放大）

- `cv2.INTER_AREA`：区域插值（适合缩小图像，可以避免波纹）

- `cv2.INTER_LANCZOS4`：Lanczos插值（8x8邻域，质量最高）

- `cv2.INTER_LINEAR_EXACT`：精确双线性插值

### 2. 图像旋转 - cv2.getRotationMatrix2D() + cv2.warpAffine()

```python
# 获取旋转矩阵
M = cv2.getRotationMatrix2D(center, angle, scale)

# 应用仿射变换
rotated = cv2.warpAffine(src, M, dsize, dst=None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=None)

# 示例：以图像中心旋转45度
h, w = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
```

getRotationMatrix2D 参数说明：

`center`：旋转中心点，元组 (x, y)

`angle`：旋转角度，正值表示逆时针旋转

`scale`：缩放因子，1.0 表示保持原大小

warpAffine 参数说明：

- `src`：输入图像

- `M`：2×3 的变换矩阵

- `dsize`：输出图像大小，元组 (width, height)

- `flags`：插值方法（同 resize）

- `borderMode`：边界填充模式

  - `cv2.BORDER_CONSTANT`：常量填充

  - `cv2.BORDER_REPLICATE`：复制边界

  - `cv2.BORDER_REFLECT`：反射

  - `cv2.BORDER_WRAP`：包裹

  - `cv2.BORDER_REFLECT_101`：对称反射

- `borderValue`：当 borderMode 为 CONSTANT 时的填充值（BGR格式）

## Q&A

1. **Q**：vscode运行程序时以当前工作目录（Current Working Directory） 执行，可能不是需要的 .py 文件所在的目录，而是项目根目录或其他位置。可能会导致文件的读取以及输出路径错误。  
**A**：使用 \_\_file\_\_ 获取脚本所在目录并在代码中指定路径

    ```python
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ```  

    或在 VS Code 中设置工作目录（.vscode/settings.json）
