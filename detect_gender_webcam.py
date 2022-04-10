from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
                    
# 加载模型
model = load_model('gender_detection-1000.h5') 

# 打开摄像头
webcam = cv2.VideoCapture(0) # 如果要识别视频文件，就将视频放到程序的同目录下，并将VideoCapture中的0改成视频名称，例如cv2.VideoCapture(./video.mp4)
    
classes = ['man','woman']

# 遍历摄像头帧
while webcam.isOpened():

    # 从webcam读取帧
    status, frame = webcam.read()

    # 人脸识别，从当前帧中裁剪出人脸
    face, confidence = cv.detect_face(frame)


    # 遍历识别到的人脸
    for idx, f in enumerate(face):

        # 得到人脸锚框的坐标
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # 画锚框
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # 裁剪人脸区域
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # 预处理
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # cv2.imshow("face_crop",face_crop)
        # cv2.waitKey(1)

        # 应用性别识别模型
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # 得到识别结果的标签
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # 在人脸锚框上画结果标签
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("gender detection", frame)

    # 按Q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
webcam.release()
cv2.destroyAllWindows()