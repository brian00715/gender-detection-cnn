# -*-coding:utf8-*-
"""
从人像图片中截取人脸图片，作为CNN训练数据集
"""
import os
import cv2
import time


# 根据输入的文件夹绝对路径，将该文件夹下的所有指定suffix的文件读取存入一个list,该list的第一个元素是该文件夹的名字
def readAllImg(path, *suffix):
    try:

        s = os.listdir(path)
        resultArray = []
        fileName = os.path.basename(path)
        resultArray.append(fileName)

        for i in s:
            if i.endswith(suffix):
                document = os.path.join(path, i)
                img = cv2.imread(document)
                # cv2.imshow("1",img)
                # cv2.waitKey(1)
                resultArray.append(img)

    except IOError:
        print("Error")

    else:
        print("读取成功")
        return resultArray

# 从源路径中读取所有图片放入一个list，然后逐一进行检查，把其中的脸扣下来，存储到目标路径中


def readPicSaveFace(sourcePath, objectPath, *suffix):
    try:
        # 读取照片,注意第一个元素是文件名
        resultArray = readAllImg(sourcePath, *suffix)

        # 对list中图片逐一进行检查,找出其中的人脸然后写到目标文件夹下
        count = 1
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        for i in resultArray:
            if type(i) != str and i is not None:
                gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    # 以时间戳和读取的排序作为文件名称
                    listStr = [str(int(time.time())), str(count)]
                    fileName = ''.join(listStr)
                    f = cv2.resize(gray[y:(y + h), x:(x + w)], (200, 200))
                    face_img_name = objectPath+'%s.jpg' % fileName
                    cv2.imwrite(face_img_name, f)
                    print("已完成人脸裁切：", face_img_name)
                    count += 1

    except IOError:
        print(IOError)

    else:
        print('Already read '+str(count-1)+' Faces to Destination '+objectPath)


if __name__ == '__main__':
    readPicSaveFace('./dataset/man/', './gender_dataset_face/man/',
                    '.jpeg', '.jpg', '.JPG', 'png', 'PNG')
    readPicSaveFace('./dataset/woman/', './gender_dataset_face/woman/',
                    '.jpeg', '.jpg', '.JPG', 'png', 'PNG')
