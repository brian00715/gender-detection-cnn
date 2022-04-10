import cv2 as cv
import numpy as np
import os

img = None
def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 600
    height_new = 800
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv.resize(image, (int(width * height_new / height), height_new))
    return img_new

def face_detect_demo():#人脸检测函数
    global img
    img = img_resize(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#把图片变成灰度图片，因为人脸的特征需要在灰度图像中查找
    #以下分别是HAAR和LBP特征数据，任意选择一种即可，注意：路径中的‘/’和‘\’是有要求的
    # 通过级联检测器 cv.CascadeClassifier，加载特征数据
    face_detector = cv.CascadeClassifier("./haarcascade_frontalface_alt.xml")
    #在尺度空间对图片进行人脸检测，第一个参数是哪个图片，第二个参数是向上或向下的尺度变化，是原来尺度的1.02倍，第三个参数是在相邻的几个人脸检测矩形框内出现就认定成人脸，这里是在相邻的5个人脸检测框内出现，如果图片比较模糊的话建议降低一点
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow("result", img)#输出结果图
    cv.waitKey(0)

files = os.listdir('./test_set/')
for img_file in files:
    img = cv.imread('./test_set/'+img_file)#图片是JPG和png都可以
    face_detect_demo()
