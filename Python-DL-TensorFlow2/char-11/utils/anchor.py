import numpy as np
import tensorflow as tf


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """
    这个函数的作用就是产生(0,0)坐标开始的基础的9个anchor框，再根据不同的放缩比和宽高比进行进一步的调整。
    本代码中对应着三种面积的大小(16*8)^2 (16*16)^2 (16*32)^2
    也就是128,256,512的平方大小，三种面积乘以三种放缩比就刚刚好是9种anchor
    :param base_size: 基础的anchor的宽和高是16的大小(stride)
    :param ratios: 宽高的比例
    :param anchor_scales: 在base_size的基础上再增加的量
    :return: |ratios| * |scales|个anchors的坐标
    """
    py = base_size / 2.
    px = base_size / 2.  # 感受野的中心坐标

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.

    return anchor_base


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """
    根据上面函数在原图的第(0,0)个特征图感受野中心位置生成的anchor_base, 在原图的特征图感受野中心生成anchors
    :param anchor_base: anchors的坐标
    :param feat_stride: 特征图缩小倍数（步长）
    :param height: 特征图高度
    :param width: 特征图宽度
    :return: 原图上所有的anchors坐标，[h * w * #anchor,4]
    """
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def _get_inside_index(anchor, H, W):
    """
    计算哪些anchor完全落在图片内部
    :param anchor:
    :param H:
    :param W:
    :return:
    """
    index_inside = np.where(
        (anchor[:, 0] >= 0) &  # y1
        (anchor[:, 1] >= 0) &  # x1
        (anchor[:, 2] <= H) &  # y2
        (anchor[:, 3] <= W)  # x2
    )[0]
    return index_inside


def bbox_iou(bbox_a, bbox_b):
    """
    计算两个（a1,4）,（a2,4）形状bbox的所有iou
    返回一个（a1,a2）形状的矩阵，
    每i行，第j列代表第一个bbox列表的第i个box与第二个bbox列表第j个box的iou

    np.prod()函数用来计算所有元素的乘积，对于有多个维度的数组可以指定轴，如axis=1指定计算每一行的乘积。
    """
    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def bbox2loc(src_bbox, dst_bbox):
    """
    计算以x,y,w,h形式的两个bbox之间的offset.
    :param src_bbox: [?, 4]可以是anchors
    :param dst_bbox: [?, 4]可以是ground truth
    :return: [?, 4] 对应每个anchors变换到gt的dx,dy,dh,dw四个参数
    """
    # anchor的x,y,w,h
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width
    # ground truth的x,y,w,h
    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    # 考虑到除法的分母是不能为0的，而且式子中log内也不能为负数，不然会直接跳出显示错误。
    # eps开始的三行将可能出现的负数和零，使用eps来替换，这样就不会出现错误了。
    # finfo函数是根据height.dtype类型来获得信息，获得符合这个类型的float型，eps是取非负的最小值。
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


class AnchorTargetCreator(object):
    """
    目的：利用每张图中bbox的真实标签来为所有任务分配ground truth！
    输入：最初生成的20000个anchor坐标、此一张图中所有的bbox的真实坐标
    输出：size为（20000，1）的正负label（其中只有128个为1，128个为0，其余都为-1
          size为（20000，4）的回归目标（所有anchor的坐标都有）

    将20000多个候选的anchor选出256个anchor进行二分类和所有的anchor进行回归位置 。为上面的预测值提供相应的真实值。选择方式如下：
    对于每一个ground truth bounding box (gt_bbox)，选择和它重叠度（IoU）最高的一个anchor作为正样本。
    对于剩下的anchor，从中选择和任意一个gt_bbox重叠度超过0.7的anchor，作为正样本，正样本的数目不超过128个。
    随机选择和gt_bbox重叠度小于0.3的anchor作为负样本。负样本和正样本的总数为256。
    对于每个anchor, gt_label 要么为1（前景），要么为0（背景），所以这样实现二分类。
    在计算回归损失的时候，只计算正样本（前景）的损失，不计算负样本的位置损失。
    """
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):

        img_H, img_W = img_size
        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]

        # argmax_ious [S,] 每个Anchor对应iou最大的bbox是第几个
        # label 每个Anchor是正样本(1)、负样本(0)还是忽略(-1)。
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)

        # compute bounding box regression targets
        # 因为anchor还是x1y1x2y2的形式，转换成delta t形式
        # [S,4]
        loc = bbox2loc(anchor, bbox[argmax_ious])
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)
        # argmax_ious [S,] 每个Anchor对应iou最大的bbox是第几个
        # max_ious 每个anchor对应iou最大bbox的iou值
        # gt_argmax_ious 对于每一个(gt_bbox)，和它IoU最高的一个anchor的位置
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)
        # 如果每个anchor对应bbox的最大iou值都小于0.3,那么这就是负样本。
        label[max_ious < self.neg_iou_thresh] = 0
        # 每个bbox对应最大iou的那些anchor为正样本
        label[gt_argmax_ious] = 1
        # 每个anchor对应bbox的最大iou值都大于0.7,那么这也是正样本
        label[max_ious >= self.pos_iou_thresh] = 1
        # 正样本不能太多
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1
        # 正+负=256。正的不够128，负的来凑。
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1
        return argmax_ious, label

    # 这个类内函数用来计算ious
    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        # [S,R] S个anchor,R个bbox
        ious = bbox_iou(anchor, bbox)
        # [S,] 每个Anchor对应iou最大的bbox是第几个
        argmax_ious = ious.argmax(axis=1)
        # 返回每个anchor对应iou最大bbox的iou值。
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        # [R,] 每个bbox对应iou最大的Anchor是第几个。
        gt_argmax_ious = ious.argmax(axis=0)
        # 返回每个bbox对应iou最大anchor的iou值
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        # 对于每一个(gt_bbox)，选择和它IoU最高的一个anchor作为正样本。
        # 正样本anchor的位置
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]
        return argmax_ious, max_ious, gt_argmax_ious


# 把前面除掉的不完全在图片以内的anchors拿回来。
def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of size count)
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data

    return ret


def loc2bbox(src_bbox, loc):
    """
    把x,y,w,h转化回x1,y1,x2,y2的形式
    src_bbox: hh*ww*9个anchors
    loc:hh*ww*9, 4
    """
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


class ProposalCreator:
    """
    # 目的：为Fast-RCNN也即检测网络提供2000个训练样本
    # 输入：RPN网络中1*1卷积输出的loc和score，以及20000个anchor坐标，原图尺寸
    # 输出：2000个训练样本rois（只是2000*4的坐标，无ground truth！）
    """

    def __init__(self, nms_thresh=0.7,
                 n_pre_nms=12000, n_post_nms=2000,
                 min_size=16):
        self.nms_thresh = nms_thresh
        self.n_pre_nms = n_pre_nms
        self.n_post_nms = n_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.0):
        # Convert anchors into proposal via bbox transformations.
        # 17100,4
        roi = loc2bbox(anchor, loc)

        # Clip predicted boxes to image.
        # y小于图片高度
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        # x小于图片宽度
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        order = tf.argsort(score, direction='DESCENDING').numpy()
        order = order.ravel()[:self.n_pre_nms]
        roi = roi[order, :]
        score = score[order]
        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).
        keep = tf.image.non_max_suppression(
            roi, score, max_output_size=self.n_post_nms, iou_threshold=self.nms_thresh)
        roi = roi[keep.numpy()]

        return roi


class ProposalTargetCreator(object):
    """
    目的：为2000个rois赋予ground truth！（严格讲挑出128个赋予ground truth！）
    输入：2000个rois、一个batch（一张图）中所有的bbox ground truth（R，4）、
          对应bbox所包含的label（R，1）（VOC2007来说20类0-19）
    输出：128个sample roi（128，4）、128个gt_roi_loc（128，4）、128个gt_roi_label（128，1）
    ProposalTargetCreator是RPN网络与ROIHead网络的过渡操作，RPN会产生大约2000个RoIs，这2000个RoIs不是都拿去训练，
    而是利用ProposalTargetCreator 选择128个RoIs用以训练。选择的规则如下：
        RoIs和gt_bboxes 的IoU大于0.5的，选择一些（比如32个）
        选择 RoIs和gt_bboxes的IoU小于等于0（或者0.1）的选择一些（比如 128-32=96个）作为负样本
        # <取消了> 为了便于训练，对选择出的128个RoIs，还对他们的gt_roi_loc 进行标准化处理（减去均值除以标准差）

    对于分类问题,直接利用交叉熵损失. 而对于位置的回归损失,一样采用Smooth_L1Loss, 只不过只对正样本计算损失.
    而且是只对正样本中的这个类别4个参数计算损失。举例来说:
    一个RoI在经过FC 84后会输出一个84维的loc向量. 如果这个RoI是负样本,则这84维向量不参与计算 L1_Loss
    如果这个RoI是正样本, 属于label K, 那么它的第 K×4, K×4+1 ，K×4+2， K×4+3 这4个数参与计算损失，其余的不参与计算损失
    """
    def __init__(self, n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, bbox, label):

        n_bbox, _ = bbox.shape
        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        gt_roi_label = np.array(gt_roi_label, dtype=np.int32)
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])

        return sample_roi, gt_roi_loc, gt_roi_label


if __name__ == '__main__':
    anchor_base = generate_anchor_base()
    print(anchor_base.shape)
    print(anchor_base[0])
    anchor = _enumerate_shifted_anchor(anchor_base, 16, 20, 20)
    print(anchor.shape)
    bbox_a = np.array([[0, 0, 4, 4], [1, 1, 2, 2]])
    bbox_b = np.array([[2, 2, 4, 4], [1, 1, 4, 4], [0, 0, 2, 2]])
    ious = bbox_iou(bbox_a, bbox_b)
    print(ious)