import os
import numpy as np
import cv2
import math
import pandas as pd
from matplotlib import pyplot as plt

# 定义最大灰度级数
gray_level = 16


def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    print("图像的高宽分别为：height,width", height, width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    print("max_gray_level:", max_gray_level)
    return max_gray_level + 1


def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape

    max_gray_level = maxGrayLevel(input)
    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    if d_x >= 0 or d_y >= 0:
        for j in range(height - d_y):
            for i in range(width - d_x):
                rows = srcdata[j][i]
                cols = srcdata[j + d_y][i + d_x]
                ret[rows][cols] += 1.0
    else:
        for j in range(height):
            for i in range(width):
                rows = srcdata[j][i]
                cols = srcdata[j + d_y][i + d_x]
                ret[rows][cols] += 1.0
    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


def feature_computer(p):
    # mean:均值
    # con:对比度反映了图像的清晰度和纹理的沟纹深浅。纹理越清晰反差越大对比度也就越大。
    # eng:熵（Entropy, ENT）度量了图像包含信息量的随机性，表现了图像的复杂程度。当共生矩阵中所有值均相等或者像素值表现出最大的随机性时，熵最大。
    # asm:角二阶矩（能量），图像灰度分布均匀程度和纹理粗细的度量。当图像纹理均一规则时，能量值较大；反之灰度共生矩阵的元素值相近，能量值较小。
    # idm:反差分矩阵又称逆方差，反映了纹理的清晰程度和规则程度，纹理清晰、规律性较强、易于描述的，值较大。
    # Auto_correlation：相关性
    mean = 0.0
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    Auto_correlation = 0.0
    std2 = 0.0
    std = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            mean += p[i][j] * i / gray_level ** 2
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            Auto_correlation += p[i][j] * i * j
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
        for i in range(gray_level):
            for j in range(gray_level):
                std2 += (p[i][j] * i - mean) ** 2
        std = np.sqrt(std2)
    return mean, Asm, Con, -Eng, Idm, Auto_correlation, std


def test(image_name):
    img = cv2.imread(image_name)
    try:
        img_shape = img.shape
    except:
        print('imread error')
        return

    # 这里如果用‘/’会报错TypeError: integer argument expected, got float
    # 其实主要的错误是因为 因为cv2.resize内的参数是要求为整数
    img = cv2.resize(img, (img_shape[1] // 2, img_shape[0] // 2), interpolation=cv2.INTER_CUBIC)
    # img = cv2.resize(img, dsize=(1000, 1000))
    # print(img.shape)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm_0 = getGlcm(img_gray, 1, 0)
    # glcm_1 = getGlcm(img_gray, 0, 1)
    # glcm_2 = getGlcm(img_gray, 1, 1)
    # glcm_3 = getGlcm(img_gray, -1, 1)
    # print(glcm_0, glcm_1, glcm_2)
    # plt.imshow(glcm_0)
    # plt.show()

    mean, asm, con, eng, idm, Auto_correlation, std = feature_computer(glcm_0)

    return [mean, asm, con, eng, idm, Auto_correlation, std]


if __name__ == '__main__':
    # result1_3 = test("G:/ZED/1-11/img_left/0001_left.jpg")

    filepath = 'G:/ZED/test/Z_sum'
    filename_list = []
    for filename in os.listdir(filepath):
        filename_list.append(filename)

    for i in range(len(filename_list)):
        file = (f'{filepath}/'+filename_list[i])
        result1_3 = test(file)
        # print(result1_3[4], result1_3[5], result1_3[3], result1_3[2], result1_3[1])
        print(f'正在执行第{i}条数据')
        # dataframe = pd.DataFrame(
        #     {'idm': result1_3[4], 'Auto_correlation': result1_3[5], 'Entropy': result1_3[3],
        #      'con': result1_3[2], 'asm': result1_3[1], 'img': filename_list[i]}, index=[0])

        dataframe = pd.DataFrame(
            {'idm': result1_3[4], 'Entropy': result1_3[3],'con': result1_3[2], 'asm': result1_3[1],
             'img': filename_list[i]}, index=[0])
        if i == 0:
            dataframe.to_csv('G:/ZED/test/con2_gray.csv', mode='a', index=False, header=True)
        else:
            dataframe.to_csv('G:/ZED/test/con2_gray.csv', mode='a', index=False, header=False)
