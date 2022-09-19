import cv2
import numpy as np

step = 5
# 图片大一点，此处尺寸大一点
kernel = (64, 64)
# cv2.imread() 方法读取图片，读入默认是uint8格式的numpy array
image = cv2.imread("proj2.png")
image = image / 255.0
# image.shape[0]: 图片高
# image.shape[1]: 图片长
# image_size = (图片高, 图片长)
image_size = image.shape[:2]
# round() 方法返回浮点数x的四舍五入值
source_size = (int(round(image_size[1] * step)), int(round(image_size[0] * step)))
target_size = (int(round(kernel[0] * step)), int(round(kernel[0] * step)))
# cv2.resize() 方法对图片进行缩放，插值方法: 双线性插值（默认设置）
sI = cv2.resize(image, source_size, interpolation=cv2.INTER_LINEAR)
sp = cv2.resize(image, source_size, interpolation=cv2.INTER_LINEAR)
# cv2.blur() 方法对图像进行均值滤波
msI = cv2.blur(sI, target_size)
msp = cv2.blur(sp, target_size)
msII = cv2.blur(sI * sI, target_size)
msIp = cv2.blur(sI * sp, target_size)
vsI = msII - msI * msI
csIp = msIp - msI * msp
recA = csIp / (vsI + 0.01)
recB = msp - recA * msI
mA = cv2.resize(recA, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
mB = cv2.resize(recB, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
gf = mA * image + mB
gf = gf * 255
gf[gf > 255] = 255
# astype() 方法进行强制类型转换
final = gf.astype(np.uint8)
# cv2.imshow() 方法可以在窗口中显示图像
cv2.imshow("image", image)
cv2.imshow("final", final)
# cv2.imwrite() 方法用于将图像保存到指定的文件
cv2.imwrite("final.png", final)
# cv2.waitKey() 函数的功能是不断刷新图像, 频率时间为delay
# 设置 waitKey(0) , 则表示程序会无限制的等待用户的按键事件
cv2.waitKey(0)
