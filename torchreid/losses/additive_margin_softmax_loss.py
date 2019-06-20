from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from torch.nn import Module, Parameter, CrossEntropyLoss
import math



def l2_norm(input,axis=1):          # only l2_norm the linear layer weight, the feature also need to be l2_norm!
	norm = torch.norm(input,2,axis,True)
	output = torch.div(input, norm)
	return output

##################################  Arcface head #############################################################

class Arcface(Module):
	def __init__(self, classnum=51332, batch_size=32):
		super(Arcface, self).__init__()
		self.classnum = classnum
		self.batch_size = batch_size
		self.ce_loss = CrossEntropyLoss()
		self.s = 64.
	def forward(self, embbedings, label):
		cos_theta, cos_theta_m= embbedings
		# nB = len(output)      Wrong!!  output [32, 751]
		nB = self.batch_size     # 32
		output = cos_theta * 1.0
		idx_ = torch.arange(0, nB, dtype=torch.long)
		output[idx_, label] = cos_theta_m[idx_, label]
		output *= self.s  # scale up in order to make softmax work, first introduced in normface
		# print(nB, output.shape, label.shape)      #32 torch.Size([32, 751]) torch.Size([32])
		return self.ce_loss(output, label)

##################################  Cosface head #############################################################

class Am_softmax(Module):
	def __init__(self, classnum=51332):
		super(Am_softmax, self).__init__()
		self.classnum = classnum
		self.ce_loss = CrossEntropyLoss()
		self.s = 50.     # 30.
	def forward(self, embbedings, label):
		cos_theta, phi = embbedings
		label_copy = label
		label = label.view(-1, 1)  # size=(B,1)
		index = cos_theta.data * 0.0  # size=(B,Classnum)
		index.scatter_(1, label.data.view(-1, 1), 1)
		index = index.byte()
		output = cos_theta * 1.0
		output[index] = phi[index]  # only change the correct predicted output
		output *= self.s  # scale up in order to make softmax work, first introduced in normface
		return self.ce_loss(output, label_copy)

#
# def l2_norm(input,axis=1):          # only l2_norm the linear layer weight, the feature also need to be l2_norm!
# 	norm = torch.norm(input,2,axis,True)
# 	output = torch.div(input, norm)
# 	return output
#
# ##################################  Arcface head #############################################################
#
# class Arcface(Module):
# 	# implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
# 	def __init__(self, embedding_size=2048, classnum=51332, s=64., m=0.5):       # default embedding_size=512
# 		super(Arcface, self).__init__()
# 		self.classnum = classnum
# 		self.kernel = Parameter(torch.Tensor(embedding_size, classnum)).cuda()
# 		# initial kernel
# 		self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
# 		self.m = m  # the margin value, default is 0.5
# 		self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
# 		self.cos_m = math.cos(m)
# 		self.sin_m = math.sin(m)
# 		self.mm = self.sin_m * m  # issue 1
# 		self.threshold = math.cos(math.pi - m)
# 		self.ce_loss = CrossEntropyLoss()
#
# 	def forward(self, embbedings, label):
# 		embbedings = l2_norm(embbedings)
# 		# weights norm
# 		nB = len(embbedings)
# 		kernel_norm = l2_norm(self.kernel, axis=0)
# 		# cos(theta+m)
# 		cos_theta = torch.mm(embbedings, kernel_norm)
# 		#         output = torch.mm(embbedings,kernel_norm)
# 		cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
# 		cos_theta_2 = torch.pow(cos_theta, 2)
# 		sin_theta_2 = 1 - cos_theta_2
# 		sin_theta = torch.sqrt(sin_theta_2)
# 		cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
# 		# this condition controls the theta+m should in range [0, pi]
# 		#      0<=theta+m<=pi
# 		#     -m<=theta<=pi-m
# 		cond_v = cos_theta - self.threshold
# 		cond_mask = cond_v <= 0
# 		keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
# 		cos_theta_m[cond_mask] = keep_val[cond_mask]
# 		output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
# 		idx_ = torch.arange(0, nB, dtype=torch.long)
# 		output[idx_, label] = cos_theta_m[idx_, label]
# 		output *= self.s  # scale up in order to make softmax work, first introduced in normface
# 		# return output
# 		return self.ce_loss(output, label)
#
# ##################################  Cosface head #############################################################
#
# class Am_softmax(Module):
# 	# implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
# 	def __init__(self, embedding_size=2048, classnum=51332):         # default embedding_size=512
# 		super(Am_softmax, self).__init__()
# 		self.classnum = classnum
# 		self.kernel = Parameter(torch.Tensor(embedding_size, classnum)).cuda()
# 		# initial kernel
# 		self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
# 		self.m = 0.35  # additive margin recommended by the paper
# 		self.s = 30.  # see normface https://arxiv.org/abs/1704.06369
# 		self.ce_loss = CrossEntropyLoss()
#
# 	def forward(self, embbedings, label):
# 		embbedings = l2_norm(embbedings)
# 		kernel_norm = l2_norm(self.kernel, axis=0)
# 		cos_theta = torch.mm(embbedings, kernel_norm)
# 		cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
# 		phi = cos_theta - self.m
# 		labell = label             # size=[B]
# 		label = label.view(-1, 1)  # size=(B,1)
# 		index = cos_theta.data * 0.0  # size=(B,Classnum)
# 		index.scatter_(1, label.data.view(-1, 1), 1)
# 		index = index.byte()
# 		output = cos_theta * 1.0
# 		output[index] = phi[index]  # only change the correct predicted output
# 		output *= self.s  # scale up in order to make softmax work, first introduced in normface
# 		# return output
# 		# print(output.shape, label.shape, labell.shape)
# 		return self.ce_loss(output, labell)