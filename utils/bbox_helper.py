import numpy as np
from _collections import namedtuple

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    """
    :param corner: Corner or np.array 4*N
    :return: Center or 4p.array N
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, abs(x1 - x2), abs(y1 - y2))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """
    :param center: Center or np.array 2*N
    :return: corner or np.array 4*N
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])


def get_axis_aligned_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 -y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y1 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2

    return cx, cy, w, h


def aug_apply(bbox, param, shape, inv=False, rd=False):
    """
    Apply augumentation
    :param bbox: origin bbox in image
    :param param: augmentation params, shift/scale
    :param shape: image shape, (h, w, (c)
    :param inv: inverse
    :param rd: round box
    :return: bbox(, param)
        bbox: augumented box
        param: real augmentation param
    """
    if not inv:
        center = corner2center(bbox)
        original_center = center
        real_param = {}
        if 'scale' in param:
            scale_x, scale_y = param['scale']
            imh, imw = shape[:2]
            h, w = center.h, center.w

            scale_x = min(scale_x, float(imw) / w)
            scale_y = min(scale_y, float(imh) / h)
            center = Center(center.x, center.y, center.w * scale_x, center.y * scale_y)

        bbox = center2corner(center)

        if 'shift' in param:
            tx, ty = param['shift']
            x1, y1, x2, y2 = bbox
            imh, imw = shape[:2]

            tx = max(-x1, min(imw - 1 - x2, tx))
            ty = max(-y1, min(imh - 1 - y2, ty))
            bbox = Corner(x1 + tx, y1 + ty, x2 + tx, y2 + ty)

        if rd:
            bbox = Corner(*map(round, bbox))

        current_center = corner2center(bbox)
        real_param['scale'] = current_center.w / original_center.w, current_center.h / original_center.h
        real_param['shift'] = current_center.x - original_center.x, current_center.y - original_center.y

        return bbox, real_param

    else:
        if 'scale' in param:
            scale_x, scale_y = param['scale']
        else:
            scale_x, scale_y = 1., 1.

        if 'shift' in param:
            tx, ty = param['shift']
        else:
            tx, ty = 0, 0

        center = corner2center(bbox)
        center = Center(center.x - tx, center.y - ty, center.w / scale_x. center.h / scale_y)
        return center2corner(center)


def IoU(rect1, rect2):
    x1, x2, x3, x4 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, tx2, tx3, tx4 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(x1, tx1)
    yy1 = np.maximum(x2, tx2)
    xx2 = np.minimum(x3, tx3)
    yy2 = np.minimum(x4, tx4)

    ww = np.maximum(0, xx1 - xx2)
    hh = np.maximum(0, yy2 - yy1)

    area1 = (x3 - x1) * (x4 - x2)
    area2 = (tx3 - tx1) * (tx4 - tx2)

    union_ = ww * hh
    res = union_ / (area1 + area2 - union_)
    return res






