from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import cross_entropy



class NPairLoss(nn.Module):

	def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True, l2_reg=0.001):
		super(NPairLoss, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon if label_smooth else 0
		self.use_gpu = use_gpu
		self.criterion = nn.CrossEntropyLoss()
		self.l2_reg = l2_reg

	def forward(self, inputs, targets):
		"""
		Args:
			inputs (torch.Tensor): prediction matrix (before softmax) with
				shape (batch_size, num_classes).
			targets (torch.LongTensor): ground truth labels with shape (batch_size).
				Each position contains the label index.
		"""
		embed_anchor = inputs[0::2]
		embed_pos = inputs[1::2]
		# target = targets[0::2]
		# print('target: ', target.min(), ' to ', target.max(), type(target), target.shape)
		embed_anchor_norm = embed_anchor.norm(dim=1)
		embed_pos_norm = embed_pos.norm(dim=1)
		simliarity_matrix = embed_anchor.mm(embed_pos.transpose(0, 1))
		N = embed_anchor.size()[0]
		target = torch.from_numpy(np.array([i for i in range(N)])).cuda()
		l2loss = (embed_anchor_norm.sum() + embed_pos_norm.sum()) / N
		# print(simliarity_matrix.shape, target.min(),target.max())
		return self.criterion(simliarity_matrix, target) + l2loss * self.l2_reg, l2loss

		# batch_size = inputs.size(0)
		# embed_anchor = inputs[0::2]     # 16*751
		# embed_pos = inputs[1::2]        # 16*751
		# target = targets[0::2]-1          # 16
		# logit = torch.matmul(embed_anchor, torch.transpose(embed_pos, 0, 1))
		# print(logit.shape, target.shape)
		# loss_sce = cross_entropy(logit, target)
		# l2_loss = sum(torch.norm(inputs, p=2, dim=1)) / batch_size
		# loss = loss_sce + self.l2_reg * l2_loss
		# return loss, l2_loss

