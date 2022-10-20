import matplotlib.pyplot as plt
from utils.data import VOC_BBOX_LABEL_NAMES
import tensorflow as tf


def vis_train(img, bbox, label, roi, roi_score, epoch):
    img = (img + 1) / 2.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)

    for i in range(len(bbox)):
        y1 = bbox[i][0]
        x1 = bbox[i][1]
        y2 = bbox[i][2]
        x2 = bbox[i][3]
        height = y2 - y1
        width = x2 - x1
        ax.add_patch(plt.Rectangle((x1, y1), width, height, fill=False, edgecolor='yellow', linewidth=2))
        ax.text(x1, y1, VOC_BBOX_LABEL_NAMES[label[i]], style='italic', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})

    keep = tf.image.non_max_suppression(
        roi, roi_score, max_output_size=30, iou_threshold=0.5)

    roi = roi[keep.numpy()]

    for i in range(len(roi)):
        y1 = roi[i][0]
        x1 = roi[i][1]
        y2 = roi[i][2]
        x2 = roi[i][3]
        height = y2 - y1
        width = x2 - x1
        ax.add_patch(plt.Rectangle((x1, y1), width, height, fill=False, edgecolor='red', linewidth=1))
    plt.savefig('epoch{}.jpg'.format(epoch))
    plt.close()
