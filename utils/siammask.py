import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.anchors import Anchors


class SiamMask(nn.Module):
    def __init__(self, anchors=None, o_sz=63, g_sz=127):
        super(SiamMask, self).__init__()
        self.anchors = anchors
        self.anchors_num = len(self.anchors['ratios'] * len(self.anchors['scales']))
        self.anchor = Anchors(anchors)
        self.features = None
        self.rpn_model = None
        self.mask_model = None
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])
        self.all_anchors = None

    def set_all_anchors(self, image_center, size):
        if not self.anchor.generate_all_anchors(image_center, size):
            return

    def feature_extractor(self, x):
        return self.feature_extractor(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask(self, template, search):
        pred_mask = self.mask_model(template, search)
        return pred_mask

    def _add_rpn_loss(self, label_cls, label_loc, label_loc_weight, label_mask, label_mask_weight, rpn_pred_cls, rpn_pred_loc, rpn_pred_mask):
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)
        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, label_loc_weight)
        rpn_loss_mask, iou_m, iou_5, iou_7 = select_mask_logistic


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    """
    :param pred_loc: [b, 4k, h, w]
    :param label_loc: [b, 4k, h, w]
    :param loss_weight: [b, k, h, w]
    :return: loc loss value
    """

    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim = 1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


def get_cls_loss(pred, label, select):
    if select.nelement() == 0:
        return pred.sum() * 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
    neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()

    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5