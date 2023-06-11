import datetime
import os
from time import sleep

import torch.optim
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_reader import MyData
from loss import HardTripletLoss, CrossEntropyLabelSmooth
from module import BaseResnet50, MyModule
from until import data_set_mapper


def train(root_dir, RGB_dir, gait_dir,
          epoch, batch_size,
          learning_rate, lr_decay_gamma, step_size,
          GPU=False,
          ever_epoch_save=10):
    # 新建当前日期文件夹
    today = datetime.date.today()
    logs_path = f"logs_train/{today}/MyModule"
    module_path = f"/mnt/data/module/{today}/MyModule"
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    if not os.path.exists(module_path):
        os.makedirs(module_path)

    # 使用GPU还是CPU
    if GPU:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # 训练数据集
    train_mapper = data_set_mapper(data_set="train")
    train_set = MyData(root_dir, RGB_dir, gait_dir, train_mapper)

    # 验证数据集
    verify_mapper = data_set_mapper(data_set="verify")
    verify_set = MyData(root_dir, RGB_dir, gait_dir, verify_mapper)

    # 加载数据集
    train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    verify = DataLoader(verify_set, batch_size=batch_size, shuffle=True, drop_last=True)

    # 实例化网络
    module = MyModule(BaseResnet50, GPU=GPU).to(device)

    # 损失函数
    loss_tri = HardTripletLoss().to(device)
    loss_xent = CrossEntropyLabelSmooth(num_classes=84, use_gpu=GPU).to(device)
    loss_MSE = nn.MSELoss()
    a = 1.5  # MSE损失的权重系数

    # 优化器
    optim = torch.optim.Adam(module.parameters(), lr=learning_rate)
    # 学习率衰减控制
    scheduler = lr_scheduler.StepLR(optim, step_size=step_size, gamma=lr_decay_gamma)

    # 设置训练网络的记录参数
    total_train_step = 0  # 记录训练的次数
    # 保存验证集损失最小的3个模型
    min_verify_loss = [1000000, 1000000, 1000000]
    min_verify_loss_epoch = [0, 0, 0]

    # 创建3个初始模型用于保存最小损失模型
    for j in range(3):
        path = f"{module_path}/myModule-min-{j + 1}.pth"
        torch.save(module, path)

    # 训练
    writer = SummaryWriter(logs_path)
    for i in range(epoch):
        pbar2 = tqdm(train, ncols=120, desc=f"epoch {i + 1}")  # 进度条
        module.train()
        for data in pbar2:
            input_data, lables = data
            encode_feature, preID, fusion_feature, decode_feature = module(input_data)

            # loss
            tri = loss_tri(encode_feature, lables)
            xent = loss_xent(preID, lables)
            AE_MSE = loss_MSE(fusion_feature, decode_feature)
            loss = tri + xent + a * AE_MSE

            # 优化器优化模型
            optim.zero_grad()
            loss.backward()
            optim.step()

            # 进度条追加信息
            d = {"tri": '%.3f' % tri.item(),
                 "xent": '%.3f' % xent.item(),
                 "AE_MSE": '%.3f' % AE_MSE.item(),
                 "loss": '%.3f' % loss.item()
                 }
            pbar2.set_postfix(d)

            total_train_step += 1
            writer.add_scalar("train_tri_loss", tri.item(), total_train_step)
            writer.add_scalar("train_xent_loss", xent.item(), total_train_step)
            writer.add_scalar("train_AE_MSE_loss", AE_MSE.item(), total_train_step)
            writer.add_scalar("train_total_loss", loss.item(), total_train_step)

        # 验证集
        total_verify_tri = 0
        total_verify_xent = 0
        total_verify_AE_MSE = 0
        total_verify_loss = 0
        verify_num = 0
        pbar3 = tqdm(verify, ncols=120, desc=f"verify ")  # 进度条
        module.eval()
        with torch.no_grad():
            for data in pbar3:
                input_data, lables = data
                features, preID, fusion_feature, decode_feature = module(input_data)
                verify_tri = loss_tri(features, lables)
                verify_xent = loss_xent(preID, lables)
                verify_AE_MSE = loss_MSE(fusion_feature, decode_feature)
                verify_loss = verify_tri + verify_xent + a * verify_AE_MSE
                total_verify_tri += verify_tri
                total_verify_xent += verify_xent
                total_verify_AE_MSE += verify_AE_MSE
                total_verify_loss += verify_loss
                verify_num += 1

        arg_verify_tri = total_verify_tri / verify_num
        arg_verify_xent = total_verify_xent / verify_num
        arg_verify_AE_MSE = total_verify_AE_MSE / verify_num
        arg_verify_loss = total_verify_loss / verify_num
        print(f'epoch {i + 1} verify：'
              f"loss_tri = {arg_verify_tri}, "
              f"loss_xent = {arg_verify_xent}, "
              f"loss_AE_MSE = {arg_verify_AE_MSE}, "
              f"loss = {arg_verify_loss}")
        writer.add_scalar("verify_tri_loss", arg_verify_tri, i + 1)
        writer.add_scalar("verify_xent_loss", arg_verify_xent, i + 1)
        writer.add_scalar("verify_AE_MSE_loss", arg_verify_AE_MSE, i + 1)
        writer.add_scalar("verify_total_loss", arg_verify_loss, i + 1)

        # 保存验证损失最小的3个模型
        if arg_verify_loss < min_verify_loss[0]:
            # 后移记录与模型
            min_verify_loss[2] = min_verify_loss[1]
            min_verify_loss_epoch[2] = min_verify_loss_epoch[1]
            min_verify_loss[1] = min_verify_loss[0]
            min_verify_loss_epoch[1] = min_verify_loss_epoch[0]
            os.remove(f"{module_path}/myModule-min-3.pth")
            os.rename(f"{module_path}/myModule-min-2.pth", f"{module_path}/myModule-min-3.pth")
            os.rename(f"{module_path}/myModule-min-1.pth", f"{module_path}/myModule-min-2.pth")
            # 保存新模型
            path = f"{module_path}/myModule-min-1.pth"
            torch.save(module, path)
            min_verify_loss[0] = arg_verify_loss
            min_verify_loss_epoch[0] = i + 1
            print(f"验证集第一小损失模型 Epoch: {i + 1}\n")
        elif min_verify_loss[0] <= arg_verify_loss < min_verify_loss[1]:
            # 后移记录与模型
            min_verify_loss[2] = min_verify_loss[1]
            min_verify_loss_epoch[2] = min_verify_loss_epoch[1]
            os.remove(f"{module_path}/myModule-min-3.pth")
            os.rename(f"{module_path}/myModule-min-2.pth", f"{module_path}/myModule-min-3.pth")
            # 保存新模型
            path = f"{module_path}/myModule-min-2.pth"
            torch.save(module, path)
            min_verify_loss[1] = arg_verify_loss
            min_verify_loss_epoch[1] = i + 1
            print(f"验证集第二小损失模型 Epoch: {i + 1}\n")
        elif min_verify_loss[1] <= arg_verify_loss < min_verify_loss[2]:
            path = f"{module_path}/myModule-min-3.pth"
            torch.save(module, path)
            min_verify_loss[2] = arg_verify_loss
            min_verify_loss_epoch[2] = i + 1
            print(f"验证集第三小损失模型 Epoch: {i + 1}\n")
        sleep(0.2)

        # 保存模型
        if (i + 1) % ever_epoch_save == 0:
            number = str(i + 1).zfill(3)
            path = f"{module_path}/myModule-{number}.pth"
            torch.save(module, path)
            print(f"模型保存至 {path}\n")
            sleep(0.2)

        # 学习率衰减
        scheduler.step()

    print(f"loss最小Epoch: {min_verify_loss_epoch}")

    writer.close()


if __name__ == '__main__':
    root_dir = "/mnt/data/CASIA-B"
    RGB_dir = "/ROI/yolov7-filtering"
    gait_dir = "/GEI/all"  # GEI
    epoch = 100
    batch_size = 56

    learning_rate = 0.0001
    lr_decay_gamma = 0.96
    step_size = 1

    GPU = True
    ever_epoch_save = 5
    test = False

    flexibleFreeze = True
    ever_epoch_unfreeze = 10
    print_inf = True

    train(root_dir, RGB_dir, gait_dir,
          epoch, batch_size,
          learning_rate, lr_decay_gamma, step_size,
          GPU=GPU,
          ever_epoch_save=ever_epoch_save)
