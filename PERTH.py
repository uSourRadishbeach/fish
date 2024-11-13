# import os
# import cv2
# import math
# import pandas as pd
# # import csv
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.spatial import Delaunay
#
# Min_area = 500
# Max_area = 999999
# kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#
#
# # 计算两点间欧氏距离
# def cal_dis(point1, point2):
#     return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
#
#
# # 求三角剖分生成三角形周长的平均值
# def cal_perimeter(triangle):
#     side1 = cal_dis(triangle[0], triangle[1])
#     side2 = cal_dis(triangle[1], triangle[2])
#     side3 = cal_dis(triangle[2], triangle[0])
#
#     perimeter = side1 + side2 + side3
#     return perimeter
#
#
# # 三角剖分效果可视化
# def Del_vis(point):
#     liag = Delaunay(point)
#
#     # plt.triplot(point[:, 0], point[:, 1], liag.simplices.copy())
#     # plt.plot(point[:, 0], point[:, 1], 'o')
#     # plt.savefig('Delaunay/del_vis.png')
#     # plt.show()
#
#     # 输出构成三角剖分的两个三角形的点索引和坐标
#     # print('每个三角面对应点的索引index', liag.simplices)    # 每个三角面对应点的索引index
#     # print('每个三角面所包含的坐标点', point[liag.simplices])    # 每个三角面所包含的坐标点
#
#     # 三角形 0 是三角形 1 的唯一邻居，它与三角形 1 的顶点 1 相反：
#     # print(liag.neighbors[1])
#     # print('第一个三角面的X坐标:', point[liag.simplices[1, 0]])  # 第一个三角面的X坐标
#     # print('第一个三角面的Y坐标:', point[liag.simplices[1, 1]])  # 第一个三角面的Y坐标
#     # print('第一个三角面的Z坐标:', point[liag.simplices[1, 2]])  # 第一个三角面的Z坐标
#
#     # 统计三角形个数，并且计算三角形的周长及平均周长
#     sum_perimeter = 0
#     num = 0
#     for idx, triangle in enumerate(point[liag.simplices]):
#         perimeter = cal_perimeter(triangle)
#         sum_perimeter += perimeter
#         num += 1
#         # print(f'Triangle{idx + 1}Perimeter:{perimeter}')
#
#     avg_perimeter = sum_perimeter / num
#
#     # 三角剖分结果可视化
#     plt.triplot(point[:, 0], point[:, 1], liag.simplices.copy())
#     plt.plot(point[:, 0], point[:, 1], 'o')
#     # plt.savefig('Delaunay/del_vis_left.png')
#     plt.title(f'FIFFB = {math.ceil(avg_perimeter)}px')
#     # plt.show()
#
#     # 分别统计大于、小于平均周长的三角形个数
#     large_tri, little_tri = 0, 0
#     for idx, triangle in enumerate(point[liag.simplices]):
#         perimeter = cal_perimeter(triangle)
#         if perimeter > avg_perimeter:
#             large_tri += 1
#         else:
#             little_tri += 1
#     # 摄食与非摄食状态下的波动变化不明显，不具有参考性
#     # tri_sum = large_tri + little_tri
#     # rate_large = large_tri / tri_sum
#     # rate_little = little_tri / tri_sum
#
#     print('三角形的平均周长为：', avg_perimeter, '\n三角形个数为:', num)
#     print('周长大于平均值的三角形个数为:', large_tri)
#     print('周长小于平均值的三角形个数为:', little_tri)
#
#     # CSV方式写入文件，尚不可用
#     # with open('avg_perimeter.csv', 'a', encoding='utf-8', newline='') as f:
#     #     csv_writer = csv.writer(f)
#     #     name = ['perimeter']
#     #     csv_writer.writerow(name)
#     #     csv_writer.writerow(avg_perimeter)
#     #     f.close()
#
#     # pd 方式将数据写入CSV文件，可用    23-12-14再次修改
#     # dataframe = pd.DataFrame({'avg_perimeter': math.ceil(avg_perimeter), }, index=[0])
#     # dataframe.to_csv('Delaunay.csv', index=False)
#
#     return avg_perimeter
#
#
# # 寻找与等分点横坐标相同的两点，并以两点的纵坐标的平均值作等分点的纵坐标
# def find_points_on_contour_with_same_x(contour, center_x):
#     """暂时无用，依然存在bug，寻找不到目标点"""
#     points = []
#     for point in contour:
#         current_point = tuple(point[0])
#
#         # 找到横坐标与质心横坐标相同的点
#         if current_point[0] == center_x:
#             points.append(current_point)
#
#             # 如果找到两个点，退出循环
#             if len(points) == 2:
#                 break
#     return points
#
#
# # 图像开运算：先腐蚀再膨胀
# def opening(l):
#     j = cv2.erode(l, kernel2, iterations=2)
#     fin = cv2.dilate(j, kernel2, iterations=2)
#     return fin
#
#
# # 图像闭运算：先膨胀再腐蚀
# def closing(z):
#     y = cv2.dilate(z, kernel2, iterations=2)
#     fin = cv2.erode(y, kernel2, iterations=1)
#     return fin
#
#
# # 图像减法
# def pic_minus(img1, img2):
#     print(img1.shape, img2.shape)
#     diffImg1 = cv2.subtract(img1, img2)
#
#     return diffImg1
#
#
# # 伽马变换、效果差，图像元素花
# def img_enhance(img):
#     # 图像归一化
#     fi = img / 255.0
#
#     # 噶玛变换
#     gamma = 0.3
#     o = np.power(fi, gamma)
#     cv2.imshow('gamma', o)
#
#
# # 线性变换增强图像对比度
# def calcGrayHist(I):
#     a = 3
#     o = float(a) * I
#     # 数据截断，大于255的数值截断为255
#     o[0 > 255] = 255
#
#     o = np.round(o)
#     o = o.astype(np.uint8)
#
#     return o
#     # rows, cols = img.shape
#     # # 存储灰度直方图
#     # grayHist = np.zeros([256], np.uint64)
#     # for r in range(rows):
#     #     for c in range(cols):
#     #         grayHist[img[r][c]] += 1
#
#
# def binary(img):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     ret, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     return binary
#
#
# def OTSU(img, GrayScale):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     assert img_gray.ndim == 2, "must input a gary_img"  # shape有几个数字, ndim就是多少
#     img_gray = np.array(img_gray).ravel().astype(np.uint8)
#     u1 = 0.0  # 背景像素的平均灰度值
#     u2 = 0.0  # 前景像素的平均灰度值
#     th = 0.0
#
#     # 总的像素数目
#     PixSum = img_gray.size
#     # 各个灰度值的像素数目
#     PixCount = np.zeros(GrayScale)
#     # 各灰度值所占总像素数的比例
#     PixRate = np.zeros(GrayScale)
#     # 统计各个灰度值的像素个数
#     for i in range(PixSum):
#         # 默认灰度图像的像素值范围为GrayScale
#         Pixvalue = img_gray[i]
#         PixCount[Pixvalue] = PixCount[Pixvalue] + 1
#
#     # 确定各个灰度值对应的像素点的个数在所有的像素点中的比例。
#     for j in range(GrayScale):
#         PixRate[j] = PixCount[j] * 1.0 / PixSum
#     Max_var = 0
#     # 确定最大类间方差对应的阈值
#     for i in range(1, GrayScale):  # 从1开始是为了避免w1为0.
#         u1_tem = 0.0
#         u2_tem = 0.0
#         # 背景像素的比列
#         w1 = np.sum(PixRate[:i])
#         # 前景像素的比例
#         w2 = 1.0 - w1
#         if w1 == 0 or w2 == 0:
#             pass
#         else:  # 背景像素的平均灰度值
#             for m in range(i):
#                 u1_tem = u1_tem + PixRate[m] * m
#             u1 = u1_tem * 1.0 / w1
#             # 前景像素的平均灰度值
#             for n in range(i, GrayScale):
#                 u2_tem = u2_tem + PixRate[n] * n
#             u2 = u2_tem / w2
#             # print(u1)
#             # 类间方差公式：G=w1*w2*(u1-u2)**2
#             tem_var = w1 * w2 * np.power((u1 - u2), 2)
#             # print(tem_var)
#             # 判断当前类间方差是否为最大值。
#             if Max_var < tem_var:
#                 Max_var = tem_var  # 深拷贝，Max_var与tem_var占用不同的内存空间。
#                 th = i
#     return th
#
#
# # 获取连通域数量，并过滤掉符合要求/连通域大小符合目标物体规格的连通域
# def conn_area(bin_img):
#     src = binary(bin_img)
#     # 膨胀操作
#     bin_clo = cv2.dilate(src, kernel2, iterations=2)
#
#     # 连通域分析
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)
#     stats = list(stats)
#
#     # 过滤掉面积小于Min_area的连通域
#     filtered_labels = [labels for labels, stat in enumerate(stats) if (Min_area <= stat[4] <= Max_area)]
#     # 创建一个新的的二值化图像，只包含过滤后的连通域
#     filtered_image = np.zeros_like(bin_img)
#     for label in filtered_labels:
#         filtered_image[labels == label] = 255
#     # stats = np.array(stats)
#     # 连通域数量num_labels
#     # print('连通域数量:', num_labels)
#     print('满足条件/符合鱼体大小的连通域个数为', len(filtered_labels))
#
#     # stats_new = []
#     # for stat in stats:
#     #     if Min_area < stat[4] <= Max_area:
#     #         stats_new.append(stat)
#     # stats_new = np.array(stats_new)
#     # 连通域信息：对应各个轮廓的x, y, width, height和面积
#     # print('连通域信息：\n [\tx \t y \t width \t height \t area]\n', stats_new)
#
#     # 连通域中心
#     # print("连通域中心:", centroids)
#
#     # 每一个像素的标签， 同一个连通域的标签是一样的
#     # print('像素标签:', labels)
#
#     # 赋予不同连通域不同的颜色
#     # output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
#     # for i in range(1, num_labels):
#     #     mask = labels == i
#     #     output[:, :, 0][mask] = np.random.randint(0, 254)
#     #     output[:, :, 1][mask] = np.random.randint(0, 254)
#     #     output[:, :, 2][mask] = np.random.randint(0, 254)
#     # filtered_image = deleteMin_area(filtered_image)
#     return filtered_image
#
#
# # 寻找连通域的质心，为构建Delaunay三角网作基础
# def barycenter(pic):
#     # img_gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
#     ret, thresh = cv2.threshold(pic, 0, 255, cv2.THRESH_OTSU)
#     '''获取轮廓最左及最右的坐标值，用于求等分点的横坐标值，已解决'''
#     _, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
#     # 找寻二值图中的全部轮廓
#     num_gather_region = 0  # 聚集区域的个数
#     num_individual_region = 0  # 非聚集区域的个数
#     area_of_gather = 0   # 全部聚集区域的面积
#     area_of_small = 0   # 非聚集区域的面积
#     Num = 1
#     depth = 0
#     Delaunay_points = []
#     slopes = 0
#
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # cv2.drawContours(test, contours, -1, (0, 255, 0), 3)
#     cv2.drawContours(pic, contours, -1, (0,0,255), 2)
#     Sum_area = 0
#     # 阈值设置
#     for contour in contours:
#         M = cv2.moments(contour)
#         Sum_area += M['m00']
#         # print(M['m00'])
#     # print(Sum_area)
#     threshold = Sum_area / 20
#     number_fish = 1  # 鱼群总数
#     num_gather = 0  # 聚集个体数
#
#     for contour in contours:
#         # 计算二值图的图像矩
#         M = cv2.moments(contour)
#
#         # '''当面积大于某值时去多等分点作该连通域的质心，现变量类型输入问题暂时存在, 23-12-1,问题解决'''
#         if math.ceil(M['m00'] / threshold) <= 2 and int(M['m00']) != 0:
#             cX = int(M['m10'] / M['m00'])
#             cY = int(M['m01'] / M['m00'])
#             depth += cY
#             cv2.circle(pic, (cX, cY), 3, (0, 0, 255), -1)
#             number_fish += 1
#
#             # 获取轮廓最左与最右两点的坐标
#             left_bottom_x = tuple(contour[contour[:, :, 0].argmin()][0])
#             right_top_x = tuple(contour[contour[:, :, 0].argmax()][0])
#             left_bottom_y = tuple(contour[contour[:, :, 1].argmax()][0])
#             right_top_y = tuple(contour[contour[:, :, 1].argmin()][0])
#
#             left_bottom = (left_bottom_x[0], left_bottom_y[1])
#             right_top = (right_top_x[0], right_top_y[1])
#
#             # 以斜率描述鱼体游泳的姿态，作为衡量摄食的指标
#             slope = abs((right_top[1] - left_bottom[1])/(right_top[0] - left_bottom[0]))
#             cv2.line(pic, left_bottom, right_top, (0, 0, 255), 1)
#             slopes += slope
#
#             text = "%d" % Num
#             cv2.putText(pic, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#             Delaunay_points.append((cX, cY))
#
#             num_individual_region += 1
#             area_of_small += M['m00']
#             # print(f'当前个体区域{Num}的面积为：{M["m00"]}')
#             Num += 1
#
#         if math.ceil(M['m00'] / threshold) > 2:
#             # 求连通域最左与最右两点的横坐标
#             leftmost_x = tuple(contour[contour[:, :, 0].argmin()][0])
#             rightmost_x = tuple(contour[contour[:, :, 0].argmax()][0])
#
#             equal = math.ceil(M['m00'] / threshold)
#             num_gather += equal
#             number_fish += equal
#             print('该处预估有', equal, '条鱼聚集')
#             num_gather_region += 1
#             area_of_gather += M['m00']
#             # print(f'当前粘连区域{Num}的面积为：{M["m00"]}')
#
#             # 获取轮廓最左与最右两点的坐标
#             left_bottom_x = tuple(contour[contour[:, :, 0].argmin()][0])
#             right_top_x = tuple(contour[contour[:, :, 0].argmax()][0])
#             left_bottom_y = tuple(contour[contour[:, :, 1].argmax()][0])
#             right_top_y = tuple(contour[contour[:, :, 1].argmin()][0])
#
#             left_bottom = (left_bottom_x[0], left_bottom_y[1])
#             right_top = (right_top_x[0], right_top_y[1])
#
#             slope = abs((right_top[1] - left_bottom[1]) / (right_top[0] - left_bottom[0])) * equal
#             # cv2.line(pic, left_bottom, right_top, (0, 0, 255), 1)
#             slopes += slope
#
#             for i in range(1, equal):
#                 # 求多等分点的横坐标
#                 cX = int((leftmost_x[0] * (equal - i) + (i * rightmost_x[0])) / equal)
#                 # cY = int(M['m01'] / M['m00'])
#                 for target_point in contour:
#                     if target_point[0][0] == cX:
#                         cY = target_point[0][1]
#                         depth += cY
#                         cv2.circle(pic, (cX, cY), 2, (0, 255, 0), 3)
#                         # number_fish += 1
#                         text = "%d" % Num
#                         cv2.putText(pic, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 254), 2)
#                         Delaunay_points.append((cX, cY))
#                         Num += 1
#                         break
#                 # depth += cY
#                 # cv2.circle(pic, (cX, cY), 2, (0, 255, 0), 3)
#                 # number_fish += 1
#                 # text = "%d" % Num
#                 # cv2.putText(pic, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 254), 2)
#                 #
#                 # Delaunay_points.append((cX, cY))
#
#
#                 # cv2.putText(pic, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
#
#     # target_contour_points = contour.reshape(-1, 2)
#     #             intersection_points = [(x, y) for x, y in target_contour_points if x == cX]
#     #             if len(intersection_points) > 0:
#     #                 intersection_points = np.array(intersection_points)
#     #                 print(intersection_points)
#     #                 cv2.circle(pic, intersection_points[0], 2, (0, 255, 0), 3)
#
#     # points = []
#     # for point in contour:
#     #     current_point = tuple(point[0])
#     #     print(current_point)
#     #     if current_point[0] == 1691:
#     #         points.append(current_point)
#     #         print(current_point)
#     # if len(points) != 0:
#     #     cv2.circle(pic, points[0], 3, (0, 255, 0), -1)
#     # cY = int((cell[0][0] + cell[0][1]) / 2)
#     # print(cell)
#     # cv2.circle(pic, cell[0], 3, (0, 255, 0), -1)
#     # cv2.circle(pic, (cX, cY2), 3, (0, 255, 0), -1)
#
#     avg_slope = slopes / number_fish
#     Delaunay_points = np.array(Delaunay_points)
#     # print('粘连鱼体的面积为：', area_of_large)
#     # print('非粘连鱼体的面积为', area_of_small)
#     # print('聚集区域在总连通域面积占比为: {:.2%}'.format(area_of_large / (area_of_large + area_of_small)))
#     # print('粘连鱼体的聚集区域个数为', num_large)
#     # print('非粘连鱼体的个数为', num_small)
#
#     # dataframe = pd.DataFrame({'rate_gather': num_gather/number_fish, 'rate_gather_region': num_gather_region/(num_gather_region + num_individual_region),
#     #                           'rate_area': area_of_gather/(area_of_gather + area_of_small), 'avg_depth': depth/number_fish},
#     #                          index=[0])
#     # dataframe.to_csv('data_fish.csv', mode='a', index=False, header=False)
#
#     rate_gather_individual = num_gather/number_fish  # 聚集的个体数所占的比例
#     bili_gather_region = num_gather_region / (num_gather_region + num_individual_region)  # 聚集区域的个数占比
#     bili_gather_area = area_of_gather / (area_of_small + area_of_gather)
#     average_depth = depth / number_fish
#
#     return pic, Delaunay_points, average_depth, rate_gather_individual, bili_gather_region, bili_gather_area, avg_slope
#
#
# # 删除连通域中包含的小面积其他连通域
# def deleteMin_area(img):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 二值化
#     ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
#     # 寻找连通域
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
#     # 计算平均面积
#     areas = list()
#     for i in range(num_labels):
#         areas.append(stats[i][-1])
#         print("轮廓%d的面积:%d" % (i, stats[i][-1]))
#
#     area_avg = np.average(areas[1:]) / 6
#     print("轮廓平均面积:", area_avg)
#
#     image_filtered = np.zeros_like(img)
#     for (i, label) in enumerate(np.unique(labels)):
#         # 如果是背景，忽略
#         if label == 0:
#             continue
#         if stats[i][-1] > area_avg:
#             image_filtered[labels == i] = 255
#
#     return image_filtered
#
#     # image = 255 - img
#     # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     # _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
#     #
#     # bin_clo = closing(binary)
#     #
#     # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)
#     # # contours, hierarchy = cv2.findContours(bin_clo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # # thresh = 0
#     # # for contour in contours:
#     # #     M = cv2.moments(contour)
#     # #     thresh += M['m00']
#     # # thresh = thresh / len(contours)
#     # # print(thresh, len(contours))
#     # filtered_labels = [label for label, stat in enumerate(stats) if stat[4] > Min_area]
#     # filtered_image = np.zeros_like(img)
#     # for label in filtered_labels:
#     #     filtered_image[labels == label] = 255
#     # # print(filtered_labels)
#     #
#     # output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
#     # for i in range(1, len(filtered_labels)):
#     #     mask = labels == i
#     #     output[:, :, 0][mask] = np.random.randint(254, 255)
#     #     output[:, :, 1][mask] = np.random.randint(254, 255)
#     #     output[:, :, 2][mask] = np.random.randint(254, 255)
#     #
#     # return 255 - output
#
#
# def area_sum(p):
#     p = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
#     image = 255 - p
#     # image = cv2.cvtColor(rev, cv2.COLOR_RGB2GRAY)
#     #
#     # bin_clo = cv2.dilate(image, kernel2, iterations=1)
#     # 取反、膨胀、过滤
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
#     filtered_labels = [label for label, stat in enumerate(stats) if stat[4] > Min_area]
#     '''np.zero_like('bin_clo') 函数内的参数不对劲，具体应该是哪一个不记得，现行结果未调用 area_sum 函数'''
#     filtered_image = np.zeros_like(image)
#     for label in filtered_labels:
#         filtered_image[labels == label] = 255
#     filtered_image = 255 - filtered_image
#     print(p.shape, filtered_image.shape)
#     turn = cv2.erode(cv2.add(p, filtered_image), kernel2)
#
#     # cv2.namedWindow('num', 0)
#     # cv2.imshow('num', filtered_image)
#
#     return turn
#
#
# # 计算个体间的距离
# def cal_dis_between_individual(image, points):
#     points = list(tuple(items) for items in list(points))
#     avg_dist = 0
#     flag = 0
#     Min_dis = [float('inf'), float("inf")]
#     nearest_dot = [None, None]
#     for i in points:
#         for j in points:
#             if i != j:
#                 distance = cal_dis(i, j)
#                 if distance < max(Min_dis):
#                     index = Min_dis.index(max(Min_dis))
#                     Min_dis[index] = distance
#                     nearest_dot[index] = j
#
#         x, y = i
#         x1, y1 = nearest_dot[0]
#         x2, y2 = nearest_dot[1]
#         # print(Min_dis)
#         avg_dist += min(Min_dis)
#         flag += 1
#         # Sum = Sum + Min_dis[0] + Min_dis[1]
#         # cv2.line(image, (x, y), (x1, y1), (0, 0, 255), 2)
#         # cv2.line(image, (x, y), (x2, y2), (0, 255, 0), 2)
#         Min_dis = [float('inf'), float("inf")]
#     avg_dist = avg_dist / flag
#     return avg_dist
#
#
# # 批量处理文件
# def batch_process():
#     filepath = 'F:/ZED/12-28/img_left/afterfeeding'
#     filename_list = []
#     for filename in os.listdir(filepath):
#         filename_list.append(filename)
#
#     for i in range(len(filename_list)):
#         img1 = cv2.imread('F:/ZED/12-28/img_left/background_left.jpg')
#         img2 = cv2.imread('F:/ZED/12-28/img_left/afterfeeding/'+filename_list[i])
#         image = pic_minus(img1, img2)
#         strengthen = calcGrayHist(image)
#         con_area = conn_area(strengthen)
#         del_area = deleteMin_area(con_area)
#         close = closing(del_area)
#         sum_area = area_sum(close)
#         pic, Del_points, avg_depth, rate_num_gather, rate_gather_region, rate_gather_area, slope = barycenter(sum_area)
#
#         # 对质心点执行三角剖分
#         avg_perimeter = Del_vis(Del_points)
#         # 计算个体间的距离(仅计算距离最近的两个个体)
#         avg_individual_dis = cal_dis_between_individual(pic, Del_points)
#         dataframe = pd.DataFrame(
#             {'avg_perimeter': avg_perimeter, 'avg_depth': avg_depth, 'rate_num_gather': rate_num_gather,
#              'rate_gather_region': rate_gather_region, 'rate_gather_area': rate_gather_area,
#              'avg_individual_dis': avg_individual_dis, 'avg_slope': slope, 'img': filename_list[i]}, index=[0])
#         dataframe.to_csv('data.csv', mode='a', index=False, header=False)
#         print(i+1, '\n')
#
#
# # 处理单独文件
# def single_process():
#     path = 'F:/ZED/12-28/img_left/afterfeeding/pic1_174_left.jpg'
#     img1 = cv2.imread('F:/ZED/12-28/img_left/background_left.jpg')
#     img2 = cv2.imread(path)
#
#     img = pic_minus(img1, img2)
#     strengthen = calcGrayHist(img)
#
#     con_area = conn_area(strengthen)
#     # final = deleteMin_area(con_area)
#     final = deleteMin_area(con_area)
#     result = closing(final)
#     sum_area = area_sum(result)
#
#     # cv2.imshow('result', final)
#     # test = cv2.imread('feeding/deleteMin_area.jpg')
#     # test = closing(test)
#
#     pic, Del_points, avg_depth, rate_num_gather, rate_gather_region, rate_gather_area, slope = barycenter(sum_area)
#     # print(Del_points)
#     # 对质心点执行三角剖分
#     avg_perimeter = Del_vis(Del_points)
#     # 计算个体间的距离(仅计算距离最近的两个个体)
#     avg_individual_dis = cal_dis_between_individual(pic, Del_points)
#
#     # dataframe = pd.DataFrame({'avg_perimeter': avg_perimeter, 'avg_depth': avg_depth, 'rate_num_gather': rate_num_gather,
#     #                           'rate_gather_region': rate_gather_region, 'rate_gather_area': rate_gather_area,
#     #                           'avg_individual_dis': avg_individual_dis, 'avg_slope': slope, 'img': path[-21:]}, index=[0])
#     # dataframe.to_csv('data.csv', mode='a', index=False, header=True)
#     print(slope)
#
#     cv2.namedWindow('center_dot', 0)
#     cv2.imshow('center_dot', sum_area)
#
#     cv2.waitKey()
#
# if __name__ == '__main__':
#     # 批量处理数据集
#     # batch_process()
#     # 处理单个数据
#     single_process()

import os
import cv2
import math
import pandas as pd
# import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

Min_area = 500
Max_area = 999999
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


# 计算两点间欧氏距离
def cal_dis(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


# 求三角剖分生成三角形周长的平均值
def cal_perimeter(triangle):
    side1 = cal_dis(triangle[0], triangle[1])
    side2 = cal_dis(triangle[1], triangle[2])
    side3 = cal_dis(triangle[2], triangle[0])

    perimeter = side1 + side2 + side3
    return perimeter


# 三角剖分效果可视化
def Del_vis(point):
    liag = Delaunay(point)

    # 统计三角形个数，并且计算三角形的周长及平均周长
    sum_perimeter = 0
    num = 0
    for idx, triangle in enumerate(point[liag.simplices]):
        perimeter = cal_perimeter(triangle)
        sum_perimeter += perimeter
        num += 1
        # print(f'Triangle{idx + 1}Perimeter:{perimeter}')

    avg_perimeter = sum_perimeter / num

    # 三角剖分结果可视化
    plt.triplot(point[:, 0], point[:, 1], liag.simplices.copy())
    plt.plot(point[:, 0], point[:, 1], 'o')
    # plt.savefig('Delaunay/del_vis_left.png')
    plt.title(f'FIFFB = {math.ceil(avg_perimeter)}px')
    # plt.show()

    # 分别统计大于、小于平均周长的三角形个数
    large_tri, little_tri = 0, 0
    for idx, triangle in enumerate(point[liag.simplices]):
        perimeter = cal_perimeter(triangle)
        if perimeter > avg_perimeter:
            large_tri += 1
        else:
            little_tri += 1
    # 摄食与非摄食状态下的波动变化不明显，不具有参考性
    # tri_sum = large_tri + little_tri
    # rate_large = large_tri / tri_sum
    # rate_little = little_tri / tri_sum

    print('三角形的平均周长为：', avg_perimeter, '\n三角形个数为:', num)
    print('周长大于平均值的三角形个数为:', large_tri)
    print('周长小于平均值的三角形个数为:', little_tri)

    return avg_perimeter


# 寻找与等分点横坐标相同的两点，并以两点的纵坐标的平均值作等分点的纵坐标
def find_points_on_contour_with_same_x(contour, center_x):
    """暂时无用，依然存在bug，寻找不到目标点"""
    points = []
    for point in contour:
        current_point = tuple(point[0])

        # 找到横坐标与质心横坐标相同的点
        if current_point[0] == center_x:
            points.append(current_point)

            # 如果找到两个点，退出循环
            if len(points) == 2:
                break
    return points


# 图像开运算：先腐蚀再膨胀
def opening(l):
    j = cv2.erode(l, kernel2, iterations=2)
    fin = cv2.dilate(j, kernel2, iterations=2)
    return fin


# 图像闭运算：先膨胀再腐蚀
def closing(z):
    y = cv2.dilate(z, kernel2, iterations=2)
    fin = cv2.erode(y, kernel2, iterations=1)
    return fin


# 图像减法
def pic_minus(img1, img2):
    print(img1.shape, img2.shape)
    diffImg1 = cv2.subtract(img1, img2)

    return diffImg1


# 伽马变换、效果差，图像元素花
def img_enhance(img):
    # 图像归一化
    fi = img / 255.0

    # 伽马变换
    gamma = 0.3
    o = np.power(fi, gamma)
    cv2.imshow('gamma', o)


# 线性变换增强图像对比度
def calcGrayHist(I):
    a = 3
    o = float(a) * I
    # 数据截断，大于255的数值截断为255
    o[0 > 255] = 255

    o = np.round(o)
    o = o.astype(np.uint8)

    return o


def binary(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def OTSU(img, GrayScale):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    assert img_gray.ndim == 2, "must input a gary_img"  # shape有几个数字, ndim就是多少
    img_gray = np.array(img_gray).ravel().astype(np.uint8)
    u1 = 0.0  # 背景像素的平均灰度值
    u2 = 0.0  # 前景像素的平均灰度值
    th = 0.0

    # 总的像素数目
    PixSum = img_gray.size
    # 各个灰度值的像素数目
    PixCount = np.zeros(GrayScale)
    # 各灰度值所占总像素数的比例
    PixRate = np.zeros(GrayScale)
    # 统计各个灰度值的像素个数
    for i in range(PixSum):
        # 默认灰度图像的像素值范围为GrayScale
        Pixvalue = img_gray[i]
        PixCount[Pixvalue] = PixCount[Pixvalue] + 1

    # 确定各个灰度值对应的像素点的个数在所有的像素点中的比例。
    for j in range(GrayScale):
        PixRate[j] = PixCount[j] * 1.0 / PixSum
    Max_var = 0
    # 确定最大类间方差对应的阈值
    for i in range(1, GrayScale):  # 从1开始是为了避免w1为0.
        u1_tem = 0.0
        u2_tem = 0.0
        # 背景像素的比列
        w1 = np.sum(PixRate[:i])
        # 前景像素的比例
        w2 = 1.0 - w1
        if w1 == 0 or w2 == 0:
            pass
        else:  # 背景像素的平均灰度值
            for m in range(i):
                u1_tem = u1_tem + PixRate[m] * m
            u1 = u1_tem * 1.0 / w1
            # 前景像素的平均灰度值
            for n in range(i, GrayScale):
                u2_tem = u2_tem + PixRate[n] * n
            u2 = u2_tem / w2
            # print(u1)
            # 类间方差公式：G=w1*w2*(u1-u2)**2
            tem_var = w1 * w2 * np.power((u1 - u2), 2)
            # print(tem_var)
            # 判断当前类间方差是否为最大值。
            if Max_var < tem_var:
                Max_var = tem_var  # 深拷贝，Max_var与tem_var占用不同的内存空间。
                th = i
    return th


# 获取连通域数量，并过滤掉符合要求/连通域大小符合目标物体规格的连通域
def conn_area(bin_img):
    src = binary(bin_img)
    # 膨胀操作
    bin_clo = cv2.dilate(src, kernel2, iterations=2)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)
    stats = list(stats)

    # 过滤掉面积小于Min_area的连通域
    filtered_labels = [labels for labels, stat in enumerate(stats) if (Min_area <= stat[4] <= Max_area)]
    # 创建一个新的的二值化图像，只包含过滤后的连通域
    filtered_image = np.zeros_like(bin_img)
    for label in filtered_labels:
        filtered_image[labels == label] = 255
    # stats = np.array(stats)
    # 连通域数量num_labels
    # print('连通域数量:', num_labels)
    # print('满足条件/符合鱼体大小的连通域个数为', len(filtered_labels))

    # stats_new = []
    # for stat in stats:
    #     if Min_area < stat[4] <= Max_area:
    #         stats_new.append(stat)
    # stats_new = np.array(stats_new)
    # 连通域信息：对应各个轮廓的x, y, width, height和面积
    # print('连通域信息：\n [\tx \t y \t width \t height \t area]\n', stats_new)

    # 连通域中心
    # print("连通域中心:", centroids)

    # 每一个像素的标签， 同一个连通域的标签是一样的
    # print('像素标签:', labels)

    # 赋予不同连通域不同的颜色
    # output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # for i in range(1, num_labels):
    #     mask = labels == i
    #     output[:, :, 0][mask] = np.random.randint(0, 254)
    #     output[:, :, 1][mask] = np.random.randint(0, 254)
    #     output[:, :, 2][mask] = np.random.randint(0, 254)
    # filtered_image = deleteMin_area(filtered_image)
    return filtered_image


# 寻找连通域的质心，为构建Delaunay三角网作基础
def barycenter(pic, threshold):
    # 调用实参为result时采用
    img_gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    #  调用实参为sum_area时采用
    # ret, thresh = cv2.threshold(pic, 0, 255, cv2.THRESH_OTSU)
    '''获取轮廓最左及最右的坐标值，用于求等分点的横坐标值，已解决'''
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    # 找寻二值图中的全部轮廓
    num_gather_region = 0  # 聚集区域的个数
    num_single = 0  # 非聚集区域的个数/单独个体的个数
    area_of_gather = 0  # 全部聚集区域的面积
    area_of_individual = 0  # 全部单独个体的面积之和
    Num = 0
    depth = []
    Delaunay_points = []
    slopes = 0

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(pic, contours, -1, (0, 0, 255), 2)
    Sum_area = 0
    # 阈值设置
    for contour in contours:
        M = cv2.moments(contour)
        Sum_area += M['m00']
        # print(M['m00'])
    # print(Sum_area)
    # threshold = Sum_area / 20
    number_fish = 0  # 鱼的总数
    num_gather = 0  # 聚集个体数

    for contour in contours:
        # 计算二值图的图像矩
        M = cv2.moments(contour)
        # 获取轮廓最左与最右两点的坐标
        left_bottom_x = tuple(contour[contour[:, :, 0].argmin()][0])
        right_top_x = tuple(contour[contour[:, :, 0].argmax()][0])
        left_bottom_y = tuple(contour[contour[:, :, 1].argmax()][0])
        right_top_y = tuple(contour[contour[:, :, 1].argmin()][0])

        # '''当面积大于某值时去多等分点作该连通域的质心，现变量类型输入问题暂时存在, 23-12-1,问题解决'''
        if math.ceil(M['m00'] / threshold) <= 2 and int(M['m00']) != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            depth.append(cY)
            cv2.circle(pic, (cX, cY), 3, (0, 255, 0), -1)
            number_fish += 1

            left_bottom = (left_bottom_x[0], left_bottom_y[1])
            right_top = (right_top_x[0], right_top_y[1])

            # 以斜率描述鱼体游泳的姿态，作为衡量摄食的指标
            slope = abs((right_top[1] - left_bottom[1]) / (right_top[0] - left_bottom[0]))
            cv2.line(pic, left_bottom, right_top, (0, 0, 255), 1)
            # print('斜率为:', slope)
            slopes += slope

            text = "%d" % Num
            cv2.putText(pic, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            Delaunay_points.append((cX, cY))

            num_single += 1
            area_of_individual += M['m00']
            # print(f'当前个体区域{Num}的面积为：{M["m00"]}')
            Num += 1

        if math.ceil(M['m00'] / threshold) > 2:
            # 求连通域最左与最右两点的横坐标
            leftmost_x = tuple(contour[contour[:, :, 0].argmin()][0])
            rightmost_x = tuple(contour[contour[:, :, 0].argmax()][0])

            equal = math.ceil(M['m00'] / threshold)
            num_gather += equal
            number_fish += equal
            print('该处预估有', equal, '条鱼聚集')
            num_gather_region += 1
            area_of_gather += M['m00']

            # 获取轮廓最左与最右两点的坐标
            left_bottom_x = tuple(contour[contour[:, :, 0].argmin()][0])
            right_top_x = tuple(contour[contour[:, :, 0].argmax()][0])
            left_bottom_y = tuple(contour[contour[:, :, 1].argmax()][0])
            right_top_y = tuple(contour[contour[:, :, 1].argmin()][0])

            left_bottom = (left_bottom_x[0], left_bottom_y[1])
            right_top = (right_top_x[0], right_top_y[1])

            # 群体姿态的斜率描述方法   前后对比不大，弃用
            slope = abs((right_top[1] - left_bottom[1]) / (right_top[0] - left_bottom[0])) * equal
            cv2.line(pic, left_bottom, right_top, (0, 0, 255), 1)
            slopes += slope

            for i in range(1, equal):
                # 求多等分点的横坐标
                cX = int((leftmost_x[0] * (equal - i) + (i * rightmost_x[0])) / equal)
                c = int(M['m01'] / M['m00'])
                cv2.circle(pic, (cX, c), 3, (0, 0, 255), -1)
                for target_point in contour:
                    # print(target_point)
                    if target_point[0][0] == cX:
                        # cY = int((target_point[0][1] + c) / 2)
                        cY = int((target_point[0][1] + c) / 2)
                        depth.append(cY)
                        # number_fish += 1
                        # cv2.circle(pic, (cX, cY), 3, (0, 0, 255), 3)

                        text = "%d" % Num
                        cv2.putText(pic, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        Delaunay_points.append((cX, cY))
                        Num += 1
                        break
    surface = 0
    for i in range(len(depth)):
        if depth[i] > np.mean(depth):
            surface += 1

    # print(depth)
    avg_slope = slopes / number_fish
    Delaunay_points = np.array(Delaunay_points)

    rate_num_gather = num_gather / number_fish  # 聚集的个体数所占的比例
    # rate_num_gather = num_gather / number_fish
    # bili_gather_region = num_gather_region / (num_gather_region + num_individual_region)  # 聚集区域的个数占比
    rate_gather_area = area_of_gather / (area_of_individual + area_of_gather)
    average_depth = np.mean(depth)
    rate_surf = surface / len(depth)
    # print(number_fish)

    return (pic, Delaunay_points, average_depth, rate_num_gather,
            rate_gather_area, area_of_gather, area_of_individual, avg_slope, rate_surf)


# 删除连通域中包含的小面积其他连通域
def filter_area(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    # 寻找连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    # 计算平均面积
    areas = list()
    for i in range(num_labels):
        areas.append(stats[i][-1])
        # print("轮廓%d的面积:%d" % (i, stats[i][-1]))

    area_avg = np.average(areas[1:]) / 3
    print("轮廓平均面积:", area_avg)

    image_filtered = np.zeros_like(img)
    for (i, label) in enumerate(np.unique(labels)):
        # 如果是背景，忽略
        if label == 0:
            continue
        if stats[i][-1] > area_avg:
            image_filtered[labels == i] = 255

    return image_filtered, area_avg * 2


def area_sum(p):
    gray = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)
    reverse = 255 - binary

    # bin_clo = cv2.dilate(image, kernel2, iterations=1)
    # 取反、膨胀、过滤
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(reverse, connectivity=8)
    image_filtered = np.zeros_like(reverse)
    for (i, label) in enumerate(np.unique(labels)):
        # 如果是背景，忽略
        if label == 0:
            continue
        if 0 < stats[i][-1] < 1126:
            image_filtered[labels == i] = 255
    Sum = cv2.add(binary, image_filtered)

    return Sum


# 计算个体间的距离
def cal_dis_between_individual(image, points):
    points = list(tuple(items) for items in list(points))
    avg_dist = 0
    flag = 0
    Min_dis = [float('inf'), float("inf")]
    nearest_dot = [None, None]
    for i in points:
        for j in points:
            if i != j:
                distance = cal_dis(i, j)
                if distance < max(Min_dis):
                    index = Min_dis.index(max(Min_dis))
                    Min_dis[index] = distance
                    nearest_dot[index] = j

        x, y = i
        x1, y1 = nearest_dot[0]
        x2, y2 = nearest_dot[1]
        # print(Min_dis)
        avg_dist += min(Min_dis)
        flag += 1
        # Sum = Sum + Min_dis[0] + Min_dis[1]
        # cv2.line(image, (x, y), (x1, y1), (0, 0, 255), 2)
        # cv2.line(image, (x, y), (x2, y2), (0, 255, 0), 2)
        Min_dis = [float('inf'), float("inf")]
    avg_dist = avg_dist / flag
    return avg_dist


# 批量处理文件
def batch_process():
    filepath = 'G:/ZED/1-16/img_left'
    filename_list = []
    for filename in os.listdir(filepath):
        filename_list.append(filename)

    for i in range(len(filename_list)):
        background = cv2.imread('G:/ZED/1-16/avg_back_left.jpg')
        origin = cv2.imread(f'{filepath}/' + filename_list[i])
        image = pic_minus(background, origin)
        strengthen = calcGrayHist(image)
        con_area = conn_area(strengthen)
        filtered_area, thresh = filter_area(con_area)
        close = closing(filtered_area)
        sum_area = area_sum(filtered_area)
        (pic, Del_points, avg_depth, rate_num_gather, rate_gather_area, area_gather, area_individual,
         slope, rate_surf) = barycenter(close, thresh)
        # 对质心点执行三角剖分
        avg_perimeter = Del_vis(Del_points)
        # 计算个体间的距离(仅计算距离最近的两个个体)
        avg_individual_dis = cal_dis_between_individual(pic, Del_points)
        PERTH = abs(avg_perimeter - avg_depth + avg_individual_dis) * (1 - rate_num_gather) * (1 - rate_gather_area)
        dataframe = pd.DataFrame(
            {'avg_perimeter': avg_perimeter, 'avg_depth': avg_depth, 'rate_num_gather': rate_num_gather,
             'rate_gather_area': rate_gather_area, 'area_of_gather': area_gather,
             'area_of_individual': area_individual, 'avg_individual_dis': avg_individual_dis,
             'avg_slope': slope, 'rate_surface': rate_surf, 'img': filename_list[i], 'PERTH': PERTH}, index=[0])
        if i == 0:
            dataframe.to_csv('G:/ZED/1-16/Perth.csv', mode='a', index=False, header=True)
        else:
            dataframe.to_csv('G:/ZED/1-16/Perth.csv', mode='a', index=False, header=False)
        print(i + 1, '\n')


# 处理单独文件
def single_process():
    path = 'G:/ZED/1-11/pic_left/0070_left.jpg'
    background = cv2.imread('G:/ZED/1-11/bg_picleft.jpg')
    origin = cv2.imread(path)

    img = pic_minus(background, origin)
    strengthen = calcGrayHist(img)

    con_area = conn_area(strengthen)
    filtered_area, thresh = filter_area(con_area)
    result = closing(filtered_area)
    sum_area = area_sum(result)

    # cv2.imshow('result', final)
    # test = cv2.imread('feeding/deleteMin_area.jpg')
    # test = closing(test)

    (pic, Del_points, avg_depth, rate_num_gather, rate_gather_area, area_gather, area_individual,
     slope, rate_surf) = barycenter(result, thresh)
    # print(Del_points)
    # 对质心点执行三角剖分
    avg_perimeter = Del_vis(Del_points)
    # 计算个体间的距离(仅计算距离最近的两个个体)
    avg_individual_dis = cal_dis_between_individual(pic, Del_points)

    print('avg_perimeter', avg_perimeter, 'avg_depth', avg_depth, 'rate_num_gather', rate_num_gather,
          'rate_gather_area', rate_gather_area, '\narea_of_gather', area_gather,
          'area_of_individual', area_individual, 'avg_individual_dis', avg_individual_dis,
          '\navg_slope', slope, 'rate_surf', rate_surf)
    # print('abg_depth:', avg_depth, '\n', 'avg_dist:', avg_individual_dis, rate_num_gather, rate_gather_area)

    cv2.namedWindow('center_dot', 0)
    cv2.imshow('center_dot', result)
    cv2.namedWindow('origin', 0)
    cv2.imshow('origin', origin)
    # cv2.namedWindow('background', 0)
    # cv2.imshow('background', background)

    cv2.waitKey()


if __name__ == '__main__':
    # 批量处理数据集
    batch_process()

    # 处理单个数据
    # single_process()
