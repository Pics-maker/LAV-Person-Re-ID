import copy
import heapq
import os
from time import sleep

import numpy as np
import torch
from tqdm import tqdm

from data_reader import MyData
from until import data_set_mapper


def split_probe_gallery_by_angle(probe_angle, gallery_angle, mapper, feature_list, ID_list):
    """
    按角度分离探针集和图库集

    :param probe_angle: 探针集中的角度
    :param gallery_angle: 图库集中的角度
    :param mapper: 对应的信息，形如 '001_nm-01_000'
    :param feature_list: 提取特征后的特征列表
    :param ID_list: 对应的ID编号
    :return: probe_set, gallery_set
    """
    # [feature, ID]
    probe_set = [[], []]
    gallery_set = [[], []]
    for ii in range(len(mapper)):
        angle = mapper[ii].split("_")[-1]
        if angle in probe_angle:
            probe_set[0].append(feature_list[ii])
            probe_set[1].append(ID_list[ii])
        if angle in gallery_angle:
            gallery_set[0].append(feature_list[ii])
            gallery_set[1].append(ID_list[ii])
    return probe_set, gallery_set


def find_min_nums(nums, find_nums):
    """
    取出列表中最小的几个数及其索引

    :param nums: 目标列表
    :param find_nums: 找几个数
    :return: min_number(最小的几个数值), min_num_index(对应索引)
    """
    if len(nums) == len(list(set(nums))):
        # 使用heapq
        min_number = heapq.nsmallest(find_nums, nums)
        min_num_index = list(map(nums.index, min_number))
    else:
        # 使用deepcopy
        nums_copy = copy.deepcopy(nums)
        max_num = max(nums) + 1
        min_num_index = []
        min_number = []
        for i in range(find_nums):
            num_min = min(nums_copy)
            num_index = nums_copy.index(num_min)
            min_number.append(num_min)
            min_num_index.append(num_index)
            nums_copy[num_index] = max_num
    return min_number, min_num_index


def get_distance_list(probe, gallery_set):
    """
    计算欧式距离，获得对应列表

    :param probe: 探针特征
    :param gallery_set: 图库/画廊
    :return: 探针特征对应每一个图库特征的距离，index与 gallery_set一致
    """
    distance_list = []
    for gallery in gallery_set:
        distance = np.linalg.norm(probe.cpu() - gallery.cpu())
        distance_list.append(distance)
    return distance_list


def Rank_data(probe_set, gallery_set, n=1):
    """
    取出每一个特征与其距离最近的前n个特征ID与其对应距离的列表

    :param probe_set: 探针集
    :param gallery_set: 图库集
    :param n: Rank-n的 n
    :return: result_ID(结果列表), result_distance(对应的距离结果)
    """
    distance_list = []  # 存储每一个特征与其他特征的距离列表 的 列表，ID对应probe_set[1]
    pbar1 = tqdm(range(len(probe_set[0])), ncols=120, desc=f"正在计算欧氏距离")  # 进度条
    for i in pbar1:
        d = get_distance_list(probe_set[0][i], gallery_set[0])
        distance_list.append(d)

    result_distance = []
    result_ID = []
    for j in range(len(probe_set[0])):
        result = distance_list[j]  # index = i 的结果，其index对应ID_list
        # 求m个最大的数值及其索引
        min_number, min_index = find_min_nums(result, n)
        # 结果——距离列表
        Rank_n_distance = min_number
        # 结果——ID列表
        Rank_n_ID = []
        for index in min_index:
            Rank_n_ID.append(gallery_set[1][index])
        # 保存结果
        result_distance.append(Rank_n_distance)
        result_ID.append(Rank_n_ID)

    return result_ID, result_distance


def calculate_Rank_n(probe_set, gallery_set, n=1):
    """
    计算Rank-n的准确率等信息

    :param probe_set: 探针集
    :param gallery_set: 图库集
    :param n: Rank-n的 n
    :return: Rank_n(Rank-n的正确率), right_not(每一个Rank-n的对错数量), result_distance(对应距离)
    """
    # 计算Rank数据
    result_ID, result_distance = Rank_data(probe_set, gallery_set, n=n)

    # 正负数结果保存列表
    right_not = []
    for _ in range(n):
        right_not.append([0, 0])

    # 计算Rank结果
    Rank_n = []  # 最终结果的列表
    assert len(result_ID) == len(probe_set[1])
    for i in range(len(result_ID)):
        for j in range(n):  # j代表Rank-n的数字n
            if result_ID[i][j] == probe_set[1][i]:  # 第i个特征距离计算最近的第j个ID与当前第i个特征的ID一致，则为正确
                right_not[j][0] += 1
            else:
                right_not[j][1] += 1
    for i in range(n):
        Rank_n.append(right_not[i][0] / (right_not[i][0] + right_not[i][1]))

    return Rank_n, right_not, result_distance


def test_rank_n(feature_list, ID_list, test_mapper, rank_n=10,
                probe_angle=["000", "090", "180"],
                gallery_angle=["018", "036", "054", "072", "108", "126", "144", "162"]):
    # 一下三个列表的index都是对应的
    # test_mapper: 对应的mapper
    # feature_list:  提取出来的特征
    # ID_list:       对应的ID列表
    # 分离probe_set, gallery_set
    probe_set, gallery_set = split_probe_gallery_by_angle(probe_angle, gallery_angle,
                                                          test_mapper, feature_list, ID_list)

    # 计算Rank-n
    Rank_n, right_not, distance_list = calculate_Rank_n(probe_set, gallery_set, n=rank_n)

    # 计算Rank-n的平均欧式距离
    average_distance = []
    for i in range(rank_n):
        average_distance.append(0)
    for i in range(len(distance_list)):
        for j in range(rank_n):
            average_distance[j] += distance_list[i][j]
    average_distance = np.array(average_distance)
    average_distance = average_distance / len(distance_list)

    return Rank_n, right_not, average_distance


def test_mAP(feature_list, ID_list, test_mapper, rank_n=10,
             probe_angle=["000", "090", "180"],
             gallery_angle=["018", "036", "054", "072", "108", "126", "144", "162"]):
    """
    计算mAP

    以下三个列表的index都是对应的
    :param feature_list: 特征列表
    :param ID_list: ID列表
    :param test_mapper: 数据信息

    :param rank_n: 计算mAP时的排序长度
    :param probe_angle: 探针角度集
    :param gallery_angle: 图库角度集
    :return:
    """
    # 分离probe_set, gallery_set
    probe_set, gallery_set = split_probe_gallery_by_angle(probe_angle, gallery_angle,
                                                          test_mapper, feature_list, ID_list)

    # 得到每个probe的匹配列表
    result_ID, _ = Rank_data(probe_set, gallery_set, n=rank_n)

    mAP = 0
    for i in range(len(result_ID)):
        AP = 0
        true = 0
        for n in range(rank_n):
            if result_ID[i][n] == probe_set[1][i]:
                true += 1
                AP += true / (n + 1)
        if true == 0:
            mAP += 0
        else:
            mAP += AP / true
    mAP /= len(result_ID)

    return mAP


if __name__ == '__main__':
    # 模型路径
    module_root_dir = "/mnt/data/module"
    data_dir = ""
    module_f = "MyModule"
    module_name = "myModule-min-1.pth"  # 用于指定哪个模型
    # 数据路径
    data_root_dir = "/mnt/data/CASIA-B"
    RGB_dir = "/ROI/yolov7-filtering"
    gait_dir = "/GEI/all"  # GEI

    # tensorboard路径
    logs_path = f"logs_test/{data_dir}"
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    # 测试什么
    mAP = True
    rank_n = True
    # 每几个Rank记录一次
    ever_rank_logs = 1

    # 构建模型路径列表
    if module_name == "":
        module_dir = f"{module_root_dir}/{data_dir}/{module_f}"
        module_list = os.listdir(module_dir)
        module_path_list = []
        for mod in module_list:
            module_path_list.append(f"{module_dir}/{mod}")
    else:  # 指定模型（只有一个模型）
        module_dir = f"{module_root_dir}/{data_dir}/{module_f}/{module_name}"
        module_path_list = [module_dir]

    # 读取测试集数据
    test_mapper = data_set_mapper(data_set="test")
    test_set = MyData(data_root_dir, RGB_dir, gait_dir, test_mapper)

    # 测试
    for m in range(len(module_path_list)):
        print(f"正在测试模型{m + 1}/{len(module_path_list)}: {module_path_list[m]}")
        # 读取模型
        module = torch.load(module_path_list[m])

        # 数据存储
        feature_list = []
        ID_list = []

        # 读取每个数据进行特征提取
        pbar = tqdm(range(len(test_set)), ncols=120, desc=f"模型正在提取特征")  # 进度条
        for j in pbar:
            data, ID = test_set[j]

            for i in range(len(data[0])):
                data[0][i] = torch.reshape(data[0][i], (1, 3, 128, 128))
            data[1] = torch.reshape(data[1], (1, 3, 128, 128))

            data[2] = torch.tensor([data[2]])
            data[3] = torch.tensor([data[3]])

            # 提取特征
            module.eval()
            with torch.no_grad():
                feature, _, _, _ = module(data)
            # 保存数据
            feature_list.append(feature)
            ID_list.append(ID)

        # 全部角度的列表
        angle_list = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]
        accuracy_rank_n = [[], [], [], [], [], [], [], [], [], [], []]
        accuracy_mAP = [[], [], [], [], [], [], [], [], [], [], []]
        # 测试角度差异
        for probe in angle_list:
            for gallery in angle_list:
                # if probe == gallery:
                #     continue
                i = int(abs((int(probe) - int(gallery))) / 18)
                print(f"\nprobe: {probe}, gallery: {gallery}")

                if rank_n:
                    Rank_n, right_not, average_distance = test_rank_n(feature_list, ID_list, test_mapper, rank_n=5,
                                                                      probe_angle=[probe],
                                                                      gallery_angle=[gallery])
                    accuracy_rank_n[i].append(Rank_n)
                    print(f'\nRank_n: {Rank_n}')
                    sleep(0.2)

                if mAP:
                    r = test_mAP(feature_list, ID_list, test_mapper, rank_n=10,
                                 probe_angle=[probe],
                                 gallery_angle=[gallery])
                    accuracy_mAP[i].append(r)
                    print(f"\nmAP: {r}")
                    sleep(0.2)

            print("\n")

        if rank_n:
            average_accuracy_rank_n = []
            for i in range(len(accuracy_rank_n)):
                average_accuracy_rank_n.append(np.mean(accuracy_rank_n[i]))
            print(f'Rank-n: {average_accuracy_rank_n}\n')

        if mAP:
            average_accuracy_mAP = []
            for i in range(len(accuracy_mAP)):
                average_accuracy_mAP.append(np.mean(accuracy_mAP[i]))
            print(f'mAP: {average_accuracy_mAP}')
