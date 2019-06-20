from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import numpy as np

from numpy.testing import assert_almost_equal

# def normalize_p(x, order=2, axis=-1):
#   """Normalizing to unit length along the specified dimension.
#   Args:
#     x: pytorch Variable
#   Returns:
#     x: pytorch Variable, same shape as input
#   """
#   x = 1. * x / (torch.norm(x, order, axis, keepdim=True).expand_as(x) + 1e-12)
#   return x

def L2Normalization(input):
	input = input.squeeze()
	return input.div(torch.norm(input, dim=1).view(-1, 1))

class HistogramLoss(nn.Module):
	def __init__(self, num_classes, num_steps = 150+1, use_gpu=True, label_smooth=True):       #num_steps : nbins 151
		super(HistogramLoss, self).__init__()
		self.num_classes = num_classes
		self.label_smooth = label_smooth
		self.step = 2 / (num_steps - 1)
		self.eps = 1 / num_steps
		self.use_gpu = use_gpu
		self.t = torch.arange(-1, 1 + self.step, self.step).view(-1, 1)
		self.tsize = self.t.size()[0]
		if self.use_gpu:
			self.t = self.t.cuda()

	def forward(self, features, classes):
		def histogram(inds, size):
			s_repeat_ = s_repeat.clone()
			indsa = (s_repeat_floor - (self.t - self.step) > -self.eps) & (
						s_repeat_floor - (self.t - self.step) < self.eps) & inds
			#assert indsa.nonzero().size()[0] == size, ('Another number of bins should be used')
			zeros = torch.zeros((1, indsa.size()[1])).byte()
			if self.cuda:
				zeros = zeros.cuda()
			indsb = torch.cat((indsa, zeros))[1:, :]
			s_repeat_[~(indsb | indsa)] = 0
			# indsa corresponds to the first condition of the second equation of the paper
			s_repeat_[indsa] = (s_repeat_ - self.t + self.step)[indsa] / self.step
			# indsb corresponds to the second condition of the second equation of the paper
			s_repeat_[indsb] = (-s_repeat_ + self.t + self.step)[indsb] / self.step

			return s_repeat_.sum(1) / size

		features = L2Normalization(features)
		classes_size = classes.size()[0]    # 64
		classes_eq = (classes.repeat(classes_size, 1) == classes.view(-1, 1).repeat(1, classes_size)).data
		dists = torch.mm(features, features.transpose(0, 1))
		assert ((dists > 1 + self.eps).sum().item() + (
					dists < -1 - self.eps).sum().item()) == 0, 'L2 normalization should be used'
		s_inds = torch.triu(torch.ones(classes_eq.size()), 1).byte()
		if self.cuda:
			s_inds = s_inds.cuda()
		pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
		neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
		pos_size = classes_eq[s_inds].sum().item()
		neg_size = (~classes_eq[s_inds]).sum().item()
		s = dists[s_inds].view(1, -1)
		s_repeat = s.repeat(self.tsize, 1)
		s_repeat_floor = (torch.floor(s_repeat.data / self.step) * self.step).float()

		histogram_pos = histogram(pos_inds, pos_size)
		#assert_almost_equal(histogram_pos.sum().item(), 1, decimal=1, err_msg='Not good positive histogram', verbose=True)
		histogram_neg = histogram(neg_inds, neg_size)
		#assert_almost_equal(histogram_neg.sum().item(), 1, decimal=1, err_msg='Not good negative histogram', verbose=True)
		histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1, histogram_pos.size()[0])
		histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.size()), -1).byte()
		if self.cuda:
			histogram_pos_inds = histogram_pos_inds.cuda()
		histogram_pos_repeat[histogram_pos_inds] = 0
		histogram_pos_cdf = histogram_pos_repeat.sum(0)
		loss = torch.sum(histogram_neg * histogram_pos_cdf)

		return loss