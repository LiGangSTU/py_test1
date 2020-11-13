import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image


# 将图像转换为灰度图
# 灰度图转换（ITU-R 601-2亮度变换）：
# L = R * 299 / 1000 + G * 587 / 1000 + B * 114 / 1000

def get_color_channels(img):
    img = img.copy()
    channels_num = len(img.shape)
    result = []
    # split函数表示将一个数组从左到右按顺序切分
    # 为什么切分图片时numpy spilt中axis参数为2?
    channels = np.split(img, channels_num, axis=2)
    for i in channels:
        result.append(i.sum(axis=2))
    # 或者 return img[:,:,0],img[:,:,1],img[:,:,2] 即可
    return result

# 读取图片
img = Image.open('img/cat.png')
img = np.array(img)
# imshow 将载入的图片显示出来 plt.imshow()函数负责对图像进行处理，并显示其格式，但是不能显示。
plt.imshow(img)
plt.show()

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 查看图片通道
print(img.shape)

# 通道分离
R,G,B = get_color_channels(img)
print(R.shape)
'''
# 生成一个白色图片，每个像素都是1.0
w = np.ones((500,500,3))
plt.imshow(w)
plt.show()

# 生成一个黑色图片，像素都是0
w1 = np.zeros(shape=(500,500,3),dtype=np.uint8)
plt.imshow(w1)
plt.show()

# 自定义颜色图片
w2 = np.full(shape=(500,500,3),fill_value=125, dtype=np.uint8)
w2[:] = [0,238,225]
plt.imshow(w2)
plt.show()

'''
# 转换为灰度图像
# imshow函数的参数 cmap将标量数据映射到色彩图 ？？？
L = R * 299/1000 + G * 587/1000 + B * 114/1000
plt.imshow(L,cmap="gray")
plt.show()

# 方法2 点积
# temp = np.array([0.299,0.587,0.114])
# plt.imshow(img@temp,cmap="gray")
# plt.show()

# 转置
plt.imshow(L.T,cmap="gray")
plt.show()

# 提取出三个通道的彩图
B_img = img.copy()
print(B_img.shape)
B_img[:,:,[0,1]] = 0
print(B_img.shape)

R_img = img.copy()
R_img[:,:,[0,2]] = 0

G_img = img.copy()
G_img[:,:,[1,2]] = 0

# 将元组分解成为fig和ax两个变量
fig,ax = plt.subplots(2,2)
ax[0,0].imshow(img)
ax[1,1].imshow(R_img)
ax[1,0].imshow(G_img)
ax[0,1].imshow(B_img)
# 指定图片输出的尺寸
fig.set_size_inches(15,15)
# tight_layout会自动调整子图参数，使之填充整个图像区域
plt.tight_layout()
plt.show()

# 图像扩展 axis 0 为列，1为行
t1 = np.concatenate((img,img,img),axis=1)
t2 = np.concatenate((t1,t1),axis=0)
plt.imshow(t2)
plt.show()

#水平镜像
m_img = img[::-1]
plt.imshow(m_img)
plt.show()

# 水平翻转
m_img_y = img[:,::-1]
plt.imshow(m_img_y)
plt.show()

# 互换xy坐标
plt.imshow(img.transpose(1,0,2))
plt.show()
plt.imshow(img.transpose(1,0,2)[::-1])
plt.show()

# 添加马赛克
mask = np.random.randint(0,256,size=(100,100,3),dtype=np.uint8)
test = img.copy()
test[100:200,200:300] = mask
plt.imshow(test)
plt.show()

# 随意打乱顺序
t = img.copy()
height = t.shape[0]
li = np.split(t,range(100,height,30),axis=0)
np.random.shuffle(li)
t = np.concatenate(li,axis=0)
plt.imshow(t)
plt.show()

# 交换通道
t1 = img.copy()
plt.imshow(t1[:,:,[2,0,1]])
plt.show()

# 交换通道
t2 = img.copy()
plt.imshow(t2[:,:,[2,1,0]])
plt.show()
