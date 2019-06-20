from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
            This criterion is a implemenation of Focal Loss, which is proposed in
            Focal Loss for Dense Object Detection.

                Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

            The losses are averaged across observations for each minibatch.
            Args:
                alpha(1D Tensor, Variable) : the scalar factor for this criterion
                gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                       putting more focus on hard, misclassiﬁed examples
                size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                    However, if the field size_average is set to False, the losses are
                                    instead summed for each minibatch.
        """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = alpha * torch.ones(class_num,1)
                self.alpha = Variable(self.alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        #print(N)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss




















    # def __init__(self, gamma=2, alpha=0.25, size_average=True): # gamma=0, alpha=None
    #     super(FocalLoss, self).__init__()
    #     self.gamma = gamma
    #     if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
    #     if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
    #     self.size_average = size_average
    #     self.logsoftmax = nn.LogSoftmax(dim=1)
    #     self.it = 0
    #
    # def forward(self, input, target):
    #     # if input.dim()>2:
    #     #     input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
    #     #     input = input.transpose(1,2)                        # N,C,H*W => N,H*W,C
    #     #     input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
    #     target = target.view(-1,1)
    #
    #     self.it += 1
    #     print('Iter: ', self.it)
    #     print(input.shape, target.shape)    # 32*751   32*1
    #     print('Target: ', target.min(), 'to', target.max(), '\n')       # 32--740
    #     # logpt = F.log_softmax(input)
    #     logpt = self.logsoftmax(input)          # [32, 751]
    #     logpt = logpt.gather(1,target)          # [32, 1]
    #     logpt = logpt.view(-1)                  # [32]
    #     pt = Variable(logpt.data.exp())         # [32]
    #     print(pt)
    #
    #     if self.alpha is not None:
    #         if self.alpha.type()!=input.data.type():    # torch.FloatTensor torch.cuda.FloatTensor
    #             self.alpha = self.alpha.type_as(input.data)
    #         # at = self.alpha.gather(0,target.data.view(-1))    # [32]     # alpha=[0.25, 0.75] 1*2      target.view(-1)  1*32
    #         # logpt = logpt * Variable(at)            # [32]
    #         # print('At.shape: ', at.shape)
    #         # print('At: ',at)
    #         # print(logpt)
    #
    #
    #     print(pt.min(), pt.max())
    #     print(logpt)
    #     loss = -1 * (1-pt)**self.gamma * logpt
    #     print(loss.mean())
    #     print('\nAAAAAAA Loss.mean(): ', loss.mean(), '\n')
    #     if self.size_average: return loss.mean()
    #     else: return loss.sum()

