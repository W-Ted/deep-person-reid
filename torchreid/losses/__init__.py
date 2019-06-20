from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss
from .A_softmax_loss import AngleLoss, AngleLoss_nomargin
from .center_loss import CenterLoss
from .n_pair_loss import NPairLoss
from .multi_similarity_loss import MultiSimilarityLoss
from .histogram_loss import HistogramLoss
from .focal_loss import FocalLoss
from .additive_margin_softmax_loss import Arcface, Am_softmax

def DeepSupervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss