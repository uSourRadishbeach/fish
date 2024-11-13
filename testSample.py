# # import cv2
# # import numpy as np
# #
# # # findContours
# # # img = cv2.imread('feeding/result.jpg')
# # # img_gary = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# # # ret, binary = cv2.threshold(img_gary, 0, 255, cv2.THRESH_OTSU)
# # #
# # # contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # # pic = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
# # #
# # # print(len(contours))
# # # cv2.namedWindow('pic', 0)
# # # cv2.imshow('pic', pic)
# # # cv2.waitKey()
# #
# # img = cv2.imread('feeding/deleteMin_area.jpg')
# #
# # image = 255 - img
# # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# # _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
# #
# # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# # bin_clo = cv2.dilate(binary, kernel2, iterations=2)
# #
# # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)
# # dataset = []
# # for i in stats:
# #     if i[4] > 100:
# #         dataset.append(i)
# #
# # dataset = np.array(dataset)
# # print(dataset)
# #
# # filtered_labels = [label for label, stat in enumerate(stats) if stat[4] > 50]
# # filtered_image = np.zeros_like(img)
# # for label in filtered_labels:
# #     filtered_image[labels == label] = 255
# #     # stats = np.array(stats)
# #     # 连通域数量num_labels
# # # print('连通域数量:', num_labels)
# # # print('满足条件/符合鱼体大小的连通域个数为', len(filtered_labels))
# # # print(stats)
# # output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
# # for i in range(1, len(filtered_labels)):
# #     mask = labels == i
# #     output[:, :, 0][mask] = np.random.randint(254, 255)
# #     output[:, :, 1][mask] = np.random.randint(254, 255)
# #     output[:, :, 2][mask] = np.random.randint(254, 255)
# #
# # result = 255 - output
# #
# # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
# # dilation = cv2.dilate(img, kernel2, iterations=2)
# # erosion = cv2.erode(result, kernel2, iterations=2)
# #
# # open_cal = cv2.dilate(erosion, kernel2, iterations=2)
# # close_cal = cv2.erode(dilation, kernel2, iterations=2)
# #
# # # cv2.namedWindow('open_cal', 0)
# # # cv2.imshow('open_cal', open_cal)
# # cv2.namedWindow('close_cal', 0)
# # cv2.imshow('close_cal', close_cal)
# # # cv2.imwrite('feeding/closing.jpg', close_cal)
# #
# # # cv2.namedWindow('dilation', 0)
# # # cv2.imshow('dilation', result)
# # # cv2.imwrite('feeding/dilation.jpg', dilation)
# # # cv2.namedWindow('erosion', 0)
# # # cv2.imshow('erosion', erosion)
# # # cv2.imwrite('feeding/latest.jpg', ker_result)
# # cv2.waitKey()
#
#
# """查找等分点且将等分点精确到轮廓之内"""
# import cv2
# import numpy as np
#
#
# def find_contours(image):
#     # 寻找图像中的连通域
#     contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours
#
#
# def divide_contour(contour, num_divisions):
#     # 获取连通域的边界框
#     x, y, w, h = cv2.boundingRect(contour)
#
#     # 计算横向等分点的步长
#     step_x = w // num_divisions
#
#     # 存储多等分点的坐标
#     division_points = []
#
#     for i in range(num_divisions):
#         # 计算当前横坐标
#         current_x = x + i * step_x
#
#         # 计算当前横坐标处连通域的中点纵坐标
#         mid_y = y + h // 2
#
#         # 添加多等分点的坐标
#         division_points.append((current_x, mid_y))
#
#     return division_points
#
#
# def main():
#     # 读取图像（假设为二值化的图像，白色为连通域）
#     image = cv2.imread('feeding/deleteMin_area.jpg', cv2.IMREAD_GRAYSCALE)
#
#     # 寻找连通域
#     contours = find_contours(image)
#     cv2.drawContours(image, contours, -1, (255, 0, 0), 3)
#     # 选择第一个连通域
#     first_contour = contours[0]
#
#     # 将连通域等分为5个点
#     num_divisions = 4
#     division_points = divide_contour(first_contour, num_divisions)
#     for i in (1, len(division_points)-1):
#         cv2.circle(image, division_points[i], 3, (255, 0, 0), 3)
#     cv2.namedWindow('visualization', 0)
#     cv2.imshow('visualization', image)
#     cv2.waitKey()
#     # 打印多等分点的坐标
#     print("Division Points:", division_points)
#
#
# if __name__ == "__main__":
#     main()
import time

# import numpy as np
# from scipy.spatial.distance import cdist
# import matplotlib.pyplot as plt
# import networkx as nx
#
#
# def find_closest_unconnected_point(node, points, connected_nodes):
#     distances = cdist([node], points)[0]
#     distances[list(connected_nodes)] = np.inf
#     closest_unconnected_point = np.argmin(distances)
#     return closest_unconnected_point
#
#
# def connect_points(points):
#     num_points = len(points)
#     graph = nx.Graph()
#     graph.add_nodes_from(range(num_points))
#
#     connected_nodes = set()
#     connections = []
#
#     for i in range(num_points):
#         if i in connected_nodes and graph.degree(i) >= 2:
#             continue
#
#         closest_unconnected_point = find_closest_unconnected_point(points[i], points, connected_nodes)
#
#         graph.add_edge(i, closest_unconnected_point)
#         connected_nodes.add(i)
#         connected_nodes.add(closest_unconnected_point)
#
#         connections.append((i, closest_unconnected_point))
#
#     return graph, connections
#
#
# def plot_points_and_connections(points, connections):
#     graph, _ = connect_points(points)
#     pos = dict(enumerate(points))
#
#     nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8)
#
#     for connection in connections:
#         point1, point2 = connection
#         plt.plot([points[point1][0], points[point2][0]], [points[point1][1], points[point2][1]], 'k-')
#
#     plt.show()
#
#
# points = [[326, 1112], [331, 998],
#           [1148, 818],
#           [436, 828],
#           [1619, 807],
#           [1065, 807],
#           [1141, 796],
#           [893, 760],
#           [984, 760],
#           [1664, 759],
#           [712, 691],
#           [312, 627]]
# # points = list(tuple(items) for items in list(points))
# print(points)
# graph, connections = connect_points(points)
# plot_points_and_connections(points, connections)
#
# # 输出连接点和线段长度
# for connection in connections:
#     point1, point2 = connection
#     distance = np.linalg.norm(points[point1] - points[point2])
#     print(f"Points {point1} and {point2} are connected. Distance: {distance}")
#
# import cv2
# import numpy as np
#
# img = cv2.imread('sample.jpg')
#
# p = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# _, binary = cv2.threshold(p, 127, 255, cv2.THRESH_OTSU)
# image = 255 - binary
#
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
#
# image_filtered = np.zeros_like(image)
# for (i, label) in enumerate(np.unique(labels)):
#     # 如果是背景，忽略
#     if label == 0:
#         continue
#     if 0 < stats[i][-1] < 5000:
#         image_filtered[labels == i] = 255
#
# Sum = cv2.add(binary, image_filtered)
# cv2.namedWindow('origin', 0)
# cv2.imshow('origin', img)
# cv2.namedWindow('reverse', 0)
# cv2.imshow('reverse', image_filtered)
# cv2.namedWindow('Sum', 0)
# cv2.imshow('Sum', Sum)
#
# cv2.waitKey()

import numpy as np

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def tetrahedron_volume(A, B, C, D):
    # 将点转换为向量
    AB = np.array(B) - np.array(A)
    AC = np.array(C) - np.array(A)
    AD = np.array(D) - np.array(A)

    # 计算体积
    volume = np.abs(np.dot(AB, np.cross(AC, AD))) / 6
    return volume

def circumsphere_radius(A, B, C, D):
    # 计算边长
    a = distance(B, C)
    b = distance(C, D)
    c = distance(D, A)
    d = distance(A, B)
    e = distance(B, D)
    f = distance(A, C)

    # 计算四面体的体积
    V = tetrahedron_volume(A, B, C, D)

    # 计算外接球半径
    R = (a * b * c) / (4 * V)
    return R

# 输入四个顶点坐标
A = (1, 0, 0)
B = (0, 1, 0)
C = (0, 0, 1)
D = (0, -1, 0)

radius = circumsphere_radius(A, B, C, D)
print(f"外接球半径: {radius:.2f}")
