import argparse
import os
import re
import cv2
import numpy as np


def gait_period(list):
    pass


def GEI(dir, view, out):
    dirlist = os.listdir(dir)
    dirlist.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))  # 对文件列表排序
    GEI = np.zeros(cv2.imread(os.path.join(dir, dirlist[0])).shape)
    for file in dirlist:
        file_path = os.path.join(dir, file)
        img = cv2.imread(file_path)
        img = img.astype(np.float32)
        GEI += img
    GEI /= len(dirlist)
    out_path = os.path.join(out, str(view))
    out_path = out_path + ".jpg"
    cv2.imwrite(out_path, GEI)
    print("save to ", out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--input_path', default='/mnt/data/CASIA-B/silhouettes-yolov7-cut-filtering', type=str,
                        help='Root path of raw dataset.')
    parser.add_argument('--output_path', default='/mnt/data/CASIA-B/GEI/all', type=str,
                        help='Root path for output.')
    opt = parser.parse_args()

    INPUT_PATH = opt.input_path
    OUTPUT_PATH = opt.output_path

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
                seq_info2 = [_id, _seq_type]
                input_dir = os.path.join(INPUT_PATH, *seq_info)
                output_dir = os.path.join(OUTPUT_PATH, *seq_info2)
                # 创建输出文件夹
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                GEI(input_dir, _view, output_dir)
