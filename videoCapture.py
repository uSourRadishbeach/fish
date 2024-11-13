import cv2

cap = cv2.VideoCapture("G:/ZED/1-23/unfeeding.avi")
c = 1
frameRate = 15  # 帧数截取间隔（每隔frameRate帧截取一帧）
Num = 1

while True:
    ret, frame = cap.read()
    if ret:
        if c % frameRate == 0:
            print("开始截取视频第：" + '{:04d}'.format(Num) + "张图片")
            # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
            cv2.imwrite("G:/ZED/1-23/unfeed/img_" + '{:04d}'.format(Num) + '.jpg', frame)  # 这里是将截取的图像保存在本地
            Num += 1
        c += 1
        cv2.waitKey(0)
    else:
        print("所有帧都已经保存完成")
        break
cap.release()
