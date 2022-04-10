from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 600
    height_new = 800
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height). height_new))
    return img_new

# 加载模型
model = load_model('gender_detection-1000.h5')
classes = ['man', 'woman']

TEST_DIR = "./test_set/"

detect_count = 1
# 遍历摄像头帧
for img_file in os.listdir(TEST_DIR):
    img = cv2.imread(TEST_DIR+img_file)
    if img is None:
            continue
    img = img_resize(img)
    
    # 人脸识别，从当前帧中裁剪出人脸
    face, confidence = cv.detect_face(img)

    # 遍历识别到的人脸
    for idx, f in enumerate(face):

        # 得到人脸锚框的坐标
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # 画锚框
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # 裁剪人脸区域
        face_crop = np.copy(img[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # 预处理
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # 应用性别识别模型
        # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
        conf = model.predict(face_crop)[0]

        # 得到识别结果的标签
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}:{:.2f}%".format(label, conf[idx] * 100)
        # label_up = "{}".format(label)
        # label_down = "{:.2f}%".format(conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10
        # 在人脸锚框上画结果标签
        cv2.putText(img, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    # 显示结果
    cv2.imwrite(TEST_DIR+"/result/"+str(detect_count)+".jpg", img)

    detect_count += 1

# 释放资源
cv2.destroyAllWindows()
