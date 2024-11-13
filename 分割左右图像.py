# coding=utf-8
import cv2
import os
# #opencv下设置双摄像头显示方法：
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#
# #opencv下分割代码方法：
# cv2.imshow('left', frame[:,0:int(w/2)])
# cv2.imshow('right', frame[:,int(w/2):])


def batch_process():
    filePath = 'G:/ZED/1-16/avg_back.jpg'

    filenamelist = []
    for file_name in os.listdir(filePath):
        print(file_name.split('.')[0])
        filenamelist.append(file_name)
    # print(filenamelist)
    # print(type(file_name.split('.')[0]))

    for i in range(len(filenamelist)):
        image = cv2.imread(f'{filePath}/' + filenamelist[i])

        # img_resize = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        x0 = 0
        y0 = 0
        size = image.shape
        width = size[1]  # 宽度
        height = size[0]  # 高度
        # 分割左右图像
        cropped_image1 = image[y0:y0 + height, x0:int(x0 + width / 2)]
        cropped_image2 = image[y0:y0 + height, int(x0 + width / 2):width]

        # 分割上下图像
        # cropped_image1 = image[y0:int(y0 + height/2), x0:x0+width]
        # cropped_image2 = image[int(y0 + height/2):y0 + height, x0: x0+width]
        # 保存裁剪后的图像
        # print(filenamelist[23].split('.')[0])
        # name_temp = filenamelist[i].split('.')[0]
        name = '{:04d}'.format(i)
        # print(name)
        # cv2.imwrite(os.path.join('D:\\imageworkspace\\testCut', nametemp+'_left.img'), cropped_image1)
        cv2.imwrite(os.path.join('G:/ZED/1-16', name + '_left.jpg'), cropped_image1)
        cv2.imwrite(os.path.join('G:/ZED/1-16', name + '_right.jpg'), cropped_image2)


def single_process():
    filename = 'G:/ZED/1-16/avg_back.jpg'
    image = cv2.imread(filename)

    # img_resize = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    x0 = 0
    y0 = 0
    size = image.shape
    width = size[1]  # 宽度
    height = size[0]  # 高度
    # 分割左右图像
    cropped_image1 = image[y0:y0 + height, x0:int(x0 + width / 2)]
    cropped_image2 = image[y0:y0 + height, int(x0 + width / 2):width]

    # 分割上下图像
    # cropped_image1 = image[y0:int(y0 + height/2), x0:x0+width]
    # cropped_image2 = image[int(y0 + height/2):y0 + height, x0: x0+width]
    # 保存裁剪后的图像
    # print(filenamelist[23].split('.')[0])
    # name_temp = filenamelist[i].split('.')[0]
    name = filename[-12:-5]
    # print(name)
    # cv2.imwrite(os.path.join('D:\\imageworkspace\\testCut', nametemp+'_left.img'), cropped_image1)
    cv2.imwrite(os.path.join('G:/ZED/1-16', name + '_left.jpg'), cropped_image1)
    cv2.imwrite(os.path.join('G:/ZED/1-16', name + '_right.jpg'), cropped_image2)

if __name__ == '__main__':
    single_process()