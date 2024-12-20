import torch
import torch.nn.functional as F
import torch.nn as nn


# def ContrastiveLoss(x, pos_x, neg_x1, neg_x2):
#     x = torch.flatten(x, start_dim=0).squeeze()
#     pos_x = torch.flatten(pos_x, start_dim=0).squeeze()
#     neg_x1 = torch.flatten(neg_x1, start_dim=0).squeeze()
#     neg_x2 = torch.flatten(neg_x2, start_dim=0).squeeze()
#     # w_1 = torch.flatten(w_1, start_dim=0).squeeze()
#     # w_2 = torch.flatten(w_2, start_dim=0).squeeze()
#
#     ##归一化矩阵样本
#     sum_x = torch.norm(x, p=1)
#     sum_pos = torch.norm(pos_x, p=1)
#     sum_neg1 = torch.norm(neg_x1, p=1)
#     sum_neg2 = torch.norm(neg_x2, p=1)
#     # sum_w_1 = torch.norm(w_1, p=1)
#     # sum_w_2 = torch.norm(w_2, p=1)
#
#     x1 = x / sum_x
#     pos_x1 = pos_x / sum_pos
#     neg_x1_1 = neg_x1 / sum_neg1
#     neg_x1_2 = neg_x2 / sum_neg2
#
#     ########################################
#     #计算样本与正负样本内积
#     pos = torch.dot(x1, pos_x1)
#     neg1 = torch.dot(x1, neg_x1_1)
#     neg2 = torch.dot(x1, neg_x1_2)
#
#     #计算正负样本得分
#     eps = 1e-13
#     pos_score = torch.exp((pos+eps)/(0.000007))
#     neg_score = torch.exp((neg1+eps)/(0.000007)) + torch.exp((neg2+eps)/(0.000007))
#     a = 0.5
#
#     #计算对比损失
#     y = (pos_score+eps) / (pos_score + a*neg_score+eps)
#     loss = (- torch.log(y+eps))
#     return loss



def ContrastiveLoss(x, pos_x, neg_x1, neg_x2, neg_x3):
    p = (x-pos_x).norm(2)
    n = (x-neg_x1).norm(2) + (x-neg_x2).norm(2) + (x-neg_x3).norm(2)

    loss = (p - (1/3)*n)*(1/(80*80*64))
    return loss
