"""
文件作用：删除yolov7截取的不完整的图片

轮廓图像保留中间的60%图片
"""

import argparse
import os
import re
import cv2


def silhouettes(dir, v, percent, out):
    dirlist = os.listdir(dir)
    dirlist.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))  # 对文件列表排序
    sum = len(dirlist)
    del_num = int(sum * (1 - percent))
    start = int(dirlist[0].split(".")[0])
    end = int(dirlist[-1].split(".")[0])
    middle = int((start + end) / 2)

    view = {
        "000": (0, 1),
        "018": (0, 1),
        "036": (0, 1),
        "054": (0.5, 0.5),
        "072": (0.5, 0.5),
        "090": (0.5, 0.5),
        "108": (0.5, 0.5),
        "126": (0.5, 0.5),
        "144": (1, 0),
        "162": (1, 0),
        "180": (1, 0),
    }

    save_start = start + int(del_num * view[v][0])
    save_end = end - int(del_num * view[v][1])
    for file in dirlist:
        if save_start <= int(file.split(".")[0]) <= save_end:
            file_path = os.path.join(dir, file)
            out_path = os.path.join(out, file)
            img = cv2.imread(file_path)
            cv2.imwrite(out_path, img)
            print("save to ", out_path)


def ROI(dir, v, percent, out):
    dirlist = os.listdir(dir)
    dirlist.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))  # 对文件列表排序
    sum = len(dirlist)
    del_num = int(sum * (1 - percent))
    start = int(dirlist[0].split("-")[0])
    end = int(dirlist[-1].split("-")[0])
    middle = int((start + end) / 2)

    view = {
        "000": (0, 1),
        "018": (0, 1),
        "036": (0, 1),
        "054": (0.5, 0.5),
        "072": (0.5, 0.5),
        "090": (0.5, 0.5),
        "108": (0.5, 0.5),
        "126": (0.5, 0.5),
        "144": (1, 0),
        "162": (1, 0),
        "180": (1, 0),
    }

    save_start = start + int(del_num * view[v][0])
    save_end = end - int(del_num * view[v][1])
    for file in dirlist:
        if save_start <= int(file.split("-")[0]) <= save_end:
            file_path = os.path.join(dir, file)
            out_path = os.path.join(out, file)
            img = cv2.imread(file_path)
            cv2.imwrite(out_path, img)
            print("save to ", out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--input_path', default='/mnt/data/CASIA-B/ROI-yolov7', type=str,
                        help='Root path of raw dataset.')
    parser.add_argument('--output_path', default='/mnt/data/CASIA-B/ROI-yolov7-filtering', type=str,
                        help='Root path for output.')
    parser.add_argument('--percent', default=0.6, type=float,
                        help='What half of the picture to keep.')
    opt = parser.parse_args()

    INPUT_PATH = opt.input_path
    OUTPUT_PATH = opt.output_path
    PERCENT = opt.percent

    # 遍历文件夹（三重循环）
    id_list = os.listdir(INPUT_PATH)
    id_list.sort()
    for _id in id_list:
        seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
        seq_type.sort()
        for _seq_type in seq_type:
            view = os.listdir(os.path.join(INPUT_PATH, _id, _seq_type))
            view.sort()
            for _view in view:
                seq_info = [_id, _seq_type, _view]
                input_dir = os.path.join(INPUT_PATH, *seq_info)
                output_dir = os.path.join(OUTPUT_PATH, *seq_info)
                # 创建输出文件夹
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # silhouettes(input_dir, _view, PERCENT, output_dir)
                ROI(input_dir, _view, PERCENT, output_dir)
