import torch
from torch import nn


class HardTripletLoss(nn.Module):
    def __init__(self, margin=2, global_feat=None, labels=None):
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        # 计算两个张量之间的相似度，两张量之间的距离>margin，loss 为正，否则loss 为 0
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        # 输入为计算的所有特征叠加起来的tensor，以及对应的标签叠加的tensor

        n = inputs.size(0)  # batch_size

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


if __name__ == '__main__':
    # # 测试TripletLoss
    # use_gpu = False
    # tri = HardTripletLoss()
    #
    # features = torch.rand(32, 2048)
    # label = torch.Tensor(
    #     [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8]).long()
    #
    # # 输入为计算的所有特征叠加起来的tensor，以及对应的标签叠加的tensor
    # loss = tri(features, label)
    # print(loss)

    # CrossEntropyLabelSmooth
    batch_size = 4  # 首先是对应的批尺寸
    num_class = 8  # 对应的分类数是8
    xent = CrossEntropyLabelSmooth(num_classes=num_class, use_gpu=False)
    input = torch.randn([batch_size, num_class])  # 这里是来随机生成一批数据的
    label = torch.randint(0, num_class - 1, [batch_size])  # 用于生成指定范围的整数，也就是对对应的标签 tensor([4, 5, 3, 3])
    print(input)
    print(label)
    print(xent(input, label))
