# import tensorflow.compat.v1 as tf
#
# tf.disable_v2_behavior()
import tensorflow as tf
import cv2
import os

size = 64
imgs = []
labs = []
model_path = "model_path"

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right


def readData(path, h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top, bottom, left, right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)

    return imgs, labs


def weightVariable(shape):
    init = tf.random.normal(shape, stddev=0.01)
    return tf.Variable(init)


def biasVariable(shape):
    init = tf.random.normal(shape)
    return tf.Variable(init)

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dropout(x, keep):
    return tf.nn.dropout(x, keep)

random_normal = tf.initializers.RandomNormal()

conv1_filters = 32
conv2_filters = 64
fc1_units = 256


def conv2d(x, W, b, strides=1):
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # tf.nn.max_pool(value, ksize, strides, padding, name=None)
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


class CNNModel(tf.Module):
    def __init__(self, name=None):
        super(CNNModel, self).__init__(name=name)
        self.w1 = tf.Variable(random_normal([3, 3, 3, conv1_filters]))  # [k_width, k_height, input_chn, output_chn]
        self.b1 = tf.Variable(tf.zeros([conv1_filters]))
        # 输入通道：32，输出通道：64，卷积后图像尺寸不变，依然是16x16
        self.w2 = tf.Variable(random_normal([3, 3, conv1_filters, conv2_filters]))
        self.b2 = tf.Variable(tf.zeros([conv2_filters]))
        # 将池第2个池化层的64个8x8的图像转换为一维的向量，长度是 64*8*8=4096
        self.w3 = tf.Variable(random_normal([4096, fc1_units]))
        self.b3 = tf.Variable(tf.zeros([fc1_units]))
        self.wout = tf.Variable(random_normal([fc1_units, 2]))
        self.bout = tf.Variable(tf.zeros([2]))

    # 正向传播
    def __call__(self, x):
        conv1 = conv2d(x, self.w1, self.b1)
        pool1 = maxpool2d(conv1, k=2)  # 将32x32图像缩小为16x16，池化不改变通道数量，因此依然是32个
        conv2 = conv2d(pool1, self.w2, self.b2)
        pool2 = maxpool2d(conv2, k=2)
        flat = tf.reshape(pool2, [-1, self.w3.get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(flat, self.w3), self.b3)
        fc1 = tf.nn.relu(fc1)
        out = tf.add(tf.matmul(fc1, self.wout), self.bout)

        return tf.nn.softmax(out)

        # 损失函数(二元交叉熵)

    # @tf.function(input_signature=[tf.TensorSpec(shape = [None,1], dtype = tf.float32),tf.TensorSpec(shape = [None,1], dtype = tf.float32)])
    def cross_entropy(self, y_pred, y_true):
        y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred)

        return tf.reduce_mean(loss_)

    # 评估指标(准确率)
    def accuracy(self, y_pred, y_true):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.reshape(tf.cast(y_true, tf.int64), [-1]))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
