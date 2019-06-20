from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch

import torchreid
from torchreid.engine import engine
from torchreid.losses import CrossEntropyLoss, MultiSimilarityLoss
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid import metrics
from torch.autograd import Variable


class ImageMultiSimilarityEngine(engine.Engine):

	def __init__(self, datamanager, model, optimizer, scheduler=None, use_cpu=False,
	             label_smooth=True):
		super(ImageMultiSimilarityEngine, self).__init__(datamanager, model, optimizer, scheduler, use_cpu)

		self.criterion = MultiSimilarityLoss(
			num_classes=self.datamanager.num_train_pids,
			use_gpu=self.use_gpu,
			label_smooth=label_smooth
		)

	def train(self, epoch, trainloader, fixbase=False, open_layers=None, print_freq=10):
		"""Trains the model for one epoch on source datasets using softmax loss.

		Args:
			epoch (int): current epoch.
			trainloader (Dataloader): training dataloader.
			fixbase (bool, optional): whether to fix base layers. Default is False.
			open_layers (str or list, optional): layers open for training.
			print_freq (int, optional): print frequency. Default is 10.
		"""
		losses = AverageMeter()
		accs = AverageMeter()
		batch_time = AverageMeter()
		data_time = AverageMeter()
		p_nums =  AverageMeter()
		n_nums = AverageMeter()

		self.model.train()

		if fixbase and (open_layers is not None):
			open_specified_layers(self.model, open_layers)
		else:
			open_all_layers(self.model)

		end = time.time()

		# print('QQ')
		# for batch_idx, data in enumerate(trainloader):
		#     imgs, pids = self._parse_data_for_train(data)
		#     print(pids)
		# print('QQ')
		# tensor([691, 691, 691, 691,  68,  68,  68,  68, 468, 468, 468, 468,  67,  67,
		#          67,  67, 232, 232, 232, 232, 293, 293, 293, 293, 244, 244, 244, 244,
		#          13,  13,  13,  13])
		# tensor([290, 290, 290, 290, 535, 535, 535, 535,  55,  55,  55,  55, 558, 558,
		#         558, 558, 129, 129, 129, 129, 699, 699, 699, 699, 232, 232, 232, 232,
		#         655, 655, 655, 655])
		# ...

		for batch_idx, data in enumerate(trainloader):
			data_time.update(time.time() - end)

			imgs, pids = self._parse_data_for_train(data)
			if self.use_gpu:
				imgs = imgs.cuda()
				pids = pids.cuda()

			self.optimizer.zero_grad()
			outputs = self.model(imgs)
			loss, p_num, n_num = self._compute_loss(self.criterion, outputs, pids)
			# loss = Variable(loss, requires_grad = True)
			if loss.item() > 0:
			  loss.backward()
			  self.optimizer.step()

			batch_time.update(time.time() - end)

			losses.update(loss.item(), pids.size(0))
			p_nums.update(p_num)
			n_nums.update(n_num)
			accs.update(metrics.accuracy(outputs, pids)[0].item())


			if (batch_idx + 1) % print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
				      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				      'MS Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				      'P-num {p.val:.2f} ({p.avg:.2f})\t'
				      'N-num {n.val:.2f} ({n.avg:.2f})\t'
				      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
					epoch + 1, batch_idx + 1, len(trainloader),
					batch_time=batch_time,
					data_time=data_time,
					loss=losses,
					p=p_nums,
					n=n_nums,
					acc=accs))

			end = time.time()

		if (self.scheduler is not None) and (not fixbase):
			self.scheduler.step()