from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import cross_entropy

def normalize(nparray, order=2, axis=0):
  """Normalize a N-D numpy array along the specified axis."""
  norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
  return nparray / (norm + np.finfo(np.float32).eps)

def normalize_p(x, order=2, axis=-1):
  """Normalizing to unit length along the specified dimension.
  Args:
    x: pytorch Variable
  Returns:
    x: pytorch Variable, same shape as input
  """
  x = 1. * x / (torch.norm(x, order, axis, keepdim=True).expand_as(x) + 1e-12)
  return x

class MultiSimilarityLoss(nn.Module):
	def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True):
		super(MultiSimilarityLoss, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon if label_smooth else 0
		self.use_gpu = use_gpu
		self.criterion = nn.CrossEntropyLoss()
		self.ep = 0.1
		self.beta = 10
		self.alpha = 2
		self.lmda = 1


	def forward(self, inputs, targets):
		"""
		Args:
			inputs (torch.Tensor): prediction matrix (before softmax) with
				shape (batch_size, num_classes).
			targets (torch.LongTensor): ground truth labels with shape (batch_size).
				Each position contains the label index.
		"""
		# inputs = normalize_p(inputs, order=2, axis=1)
		f = inputs[0::5]
		# pf = torch.delete(inputs.detach().cpu().numpy(), np.arange(0,inputs.shape[0],5),axis=0)
		pf = inputs

		nf_data = pf  # 128*512
		nf_t = nf_data.transpose(0, 1)  # 512*128
		sim_matrix = torch.mm(f.data, nf_t).squeeze()  # 32*128
		loss = torch.Tensor([0]).cuda()
		p_num = 0
		n_num = 0
		for k in range(f.shape[0]):
			loss_p = torch.Tensor([0]).cuda()
			loss_n = torch.Tensor([0]).cuda()
			sim_vector = sim_matrix[k, :]
			pos_vector = sim_vector[5*k+1: 5*k+4+1]
			neg_vector = torch.cat((sim_vector[0:5*k], sim_vector[5*k+4+1:]), 0)
			pos_min = torch.min(pos_vector)
			neg_max = torch.max(neg_vector)

			for pos_sim in pos_vector:
				if pos_sim < neg_max + self.ep:
					loss_p += torch.exp(-1 * self.alpha * (pos_sim - self.lmda))
					p_num+=1
				#print('loss_p: ',loss_p)
				# print(pos_sim)
			loss_p = torch.log(loss_p + 1) / self.alpha

			for neg_sim in neg_vector:
				if neg_sim > pos_min - self.ep:
					loss_n += torch.exp(self.beta * (neg_sim - self.lmda))
					n_num+=1
				#print('loss_n: ', loss_n)
			loss_n = torch.log(loss_n + 1) / self.beta
			# print('loss_p: ', loss_p, 'loss_n: ', loss_n)
			loss = loss_p + loss_n + loss

		return loss / f.shape[0], p_num/(4*f.shape[0]), n_num/((inputs.shape[0]-5)*f.shape[0])

