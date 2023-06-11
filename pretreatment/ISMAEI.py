import argparse
import os
import re
import cv2
import numpy as np


def AEI(dir, view, out):
    dirlist = os.listdir(dir)
    dirlist.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))  # 对文件列表排序
    AEI = np.zeros(cv2.imread(os.path.join(dir, dirlist[0])).shape)
    img_t_1 = np.zeros(cv2.imread(os.path.join(dir, dirlist[0])).shape)
    t = 0
    for file in dirlist:
        file_path = os.path.join(dir, file)
        img = cv2.imread(file_path)
        img = img.astype(np.float32)
        if t == 0:
            AEI += img
        else:
            AEI += abs(img - img_t_1)
        img_t_1 = img
        t += 1
    AEI /= len(dirlist)
    out_path = os.path.join(out, str(view))
    out_path = out_path + ".jpg"
    cv2.imwrite(out_path, AEI)
    print("AEI save to ", out_path)


def ISMAEI_same(dir, view, out):
    file_path = os.path.join(dir, view) + ".jpg"
    img = cv2.imread(file_path)
    img[int(img.shape[0] * 0.125): int(img.shape[0] * 0.625), :] = 0
    out_path = os.path.join(out, str(view))
    out_path = out_path + ".jpg"
    cv2.imwrite(out_path, img)
    print("ISMAEI save to ", out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--input_path',
                        default='/mnt/data/CASIA-B/silhouettes/silhouettes-yolov7-cut-filtering',
                        type=str,
                        help='Root path of raw dataset.')
    parser.add_argument('--AEI_output_path', default='/mnt/data/CASIA-B/AEI/original',
                        type=str,
                        help='Root path for output.')
    parser.add_argument('--ISMAEI_output_path', default='/mnt/data/CASIA-B/AEI/ISMAEI',
                        type=str,
                        help='Root path for output.')
    opt = parser.parse_args()

    INPUT_PATH = opt.input_path
    AEI_OUTPUT_PATH = opt.AEI_output_path
    ISMAEI_OUTPUT_PATH = opt.ISMAEI_output_path

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
                AEI_output_dir = os.path.join(AEI_OUTPUT_PATH, *seq_info2)
                ISMAEI_output_dir = os.path.join(ISMAEI_OUTPUT_PATH, *seq_info2)

                if not os.path.exists(ISMAEI_output_dir):
                    os.makedirs(ISMAEI_output_dir)
                ISMAEI_same(os.path.join(AEI_OUTPUT_PATH, *seq_info2), _view, ISMAEI_output_dir)
