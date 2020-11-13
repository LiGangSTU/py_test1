import numpy as np
import cv2


# 定义分离三通道函数
def get_red(img):
    return img[:, :, 2]


def get_blue(img):
    return img[:, :, 0]


def get_green(img):
    return img[:, :, 1]


if __name__ == '__main__':
    img = cv2.imread("img/cat.png")
    b, g, r = cv2.split(img)
    cv2.imshow("Blue 1", b)
    cv2.imshow("Green 1", g)
    cv2.imshow("Red 1", r)
    b = get_blue(img)
    g = get_green(img)
    r = get_red(img)
    cv2.imshow("Blue 2", b)
    cv2.imshow("Green 2", g)
    cv2.imshow("Red 2", r)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
