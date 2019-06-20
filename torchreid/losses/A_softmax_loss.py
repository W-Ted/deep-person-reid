from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class AngleLoss(nn.Module):
    def __init__(self, num_classes, gamma=0, epsilon=0.1, use_gpu=True, label_smooth=True):
        super(AngleLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        # if self.use_gpu:
        #     target = target.cuda()
        # target = (1 - self.epsilon) * target + self.epsilon / self.num_classes
        index = index.byte()
        index = Variable(index)

        # if self.it > 404*30:
        #     # self.lamb = 0
        #     output = cos_theta * 1.0  # size=(B,Classnum)
        #     output[index] -= cos_theta[index]*(1.0+0)
        #     output[index] += phi_theta[index]*(1.0+0)
        # else:
        #     # self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.01*self.it ))
        #     output = cos_theta * 1.0  # size=(B,Classnum)

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.001 * self.it))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        # logpt = F.log_softmax(output)
        logpt = self.logsoftmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss



        # W/O lambda        cos_theta/no margin
        # cos_theta, phi_theta = input
        # log_probs = self.logsoftmax(phi_theta)
        # target = torch.zeros(log_probs.size()).scatter_(1, target.unsqueeze(1).data.cpu(), 1)
        #
        # if self.use_gpu: target = target.cuda()
        # target = (1 - self.epsilon) * target + self.epsilon / self.num_classes
        # return (- target * log_probs).mean(0).sum()



        # W/O lambda        cos_m_theta/with margin
        # if self.use_gpu:
        #     target = target.cuda()
        # target = (1 - self.epsilon) * target + self.epsilon / self.num_classes

        #######
        # cos_theta,phi_theta = input
        # target = target.view(-1,1) #size=(B,1)
        # index = cos_theta.data * 0.0 #size=(B,Classnum)
        # index.scatter_(1,target.data.view(-1,1),1)
        # index = index.byte()
        # index = Variable(index)
        # self.lamb = 0.0     # max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        # output = cos_theta * 1.0 #size=(B,Classnum)
        # output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        # output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)
        #
        # log_probs = self.logsoftmax(output)
        # target = torch.zeros(log_probs.size()).scatter_(1, target.unsqueeze(1).data.cpu(), 1)
        # if self.use_gpu: target = target.cuda()
        # target = (1 - self.epsilon) * target + self.epsilon / self.num_classes
        # return (- target * log_probs).mean(0).sum()
        #######

        # logpt = F.log_softmax(output)
        # logpt = logpt.gather(1,target)
        # logpt = logpt.view(-1)
        # pt = Variable(logpt.data.exp())
        # loss = -1 * (1-pt)**self.gamma * logpt
        # loss = loss.mean()
        # return loss





        # New 1
        # self.it += 1
        # cos_theta, phi_theta = input
        # log_probs = self.logsoftmax(phi_theta)
        # target = torch.zeros(log_probs.size()).scatter_(1, target.unsqueeze(1).data.cpu(), 1)
        #
        # mask = target - 1
        # mask[mask < 0] = 1
        # self.lamb = 0
        # # self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        # mod1 = cos_theta*mask.cuda() + phi_theta*(1.0+0)/(1+self.lamb)*target.cuda() # -  cos_theta*(1.0+0)/(1+self.lamb)*target.cuda()
        # mod2 = cos_theta
        # # print((mod1-mod2).norm(p=0))
        # log_probs = self.logsoftmax(mod1)
        #
        # if self.use_gpu: target = target.cuda()
        # target = (1 - self.epsilon) * target + self.epsilon / self.num_classes
        # return (- target * log_probs).mean(0).sum()


        # New 2
        # cos_theta, phi_theta = input
        # output = cos_theta * 1.0 #size=(B,Classnum)
        # target = target.view(-1, 1)  # size=(B,1)
        #
        # logpt = F.log_softmax(output)
        # logpt = logpt.gather(1,target)
        # logpt = logpt.view(-1)
        # pt = Variable(logpt.data.exp())
        #
        # loss = -1 * (1-pt)**self.gamma * logpt
        # loss = loss.mean()
        #
        # return loss



class AngleLoss_nomargin(nn.Module):
    def __init__(self, num_classes, gamma=0, epsilon=0.1, use_gpu=True, label_smooth=True):
        super(AngleLoss_nomargin, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)
        output = cos_theta * 1.0 #size=(B,Classnum)
        logpt = self.logsoftmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()
        return loss

        # cos_theta, phi_theta = input
        # log_probs = self.logsoftmax(phi_theta)
        # target = torch.zeros(log_probs.size()).scatter_(1, target.unsqueeze(1).data.cpu(), 1)
        # if self.use_gpu: target = target.cuda()
        # target = (1 - self.epsilon) * target + self.epsilon / self.num_classes
        # return (- target * log_probs).mean(0).sum()