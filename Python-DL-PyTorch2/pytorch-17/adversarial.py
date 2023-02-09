import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models.inception import inception_v3

from PIL import Image
from scipy.misc import imsave
from scipy.interpolate import spline

import matplotlib.pyplot as plt
import os
import numpy as np
from random import randint


classes = eval(open('classes.txt').read())
trans = T.Compose([T.ToTensor(), T.Lambda(lambda t: t.unsqueeze(0))])
reverse_trans = lambda x: np.asarray(T.ToPILImage()(x))

eps = 10  # 0.01
norm = float('inf')
steps = 40
step_alpha = 0.01  # 1

model = inception_v3(pretrained=True, transform_input=True).cpu()
loss = nn.CrossEntropyLoss()
model.eval();


def load_image(img_path):
    img = trans(Image.open(img_path).convert('RGB'))
    return img


def get_class(img):
    x = Variable(img, volatile=True).cpu()
    cls= model(x).data.max(1)[1].cpu().numpy()[0]
    # cls,prediction=torch.max(model(x).data,1)
    print("cls:",cls)
    # print("prediction:",prediction)
    return classes[cls]


def get_top_five(img):
    x = Variable(img, volatile=True).cpu()
    top5 = model(x).data.topk(5)[1].cpu().numpy()[0]
    return [classes[cls] for cls in top5]


def draw_result(img, noise, adv_img):
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    orig_class, attack_class = get_class(img), get_class(adv_img)
    ax[0].imshow(reverse_trans(img[0]))
    ax[0].set_title('Original image: {}'.format(orig_class.split(',')[0]))
    ax[1].imshow(noise[0].cpu().numpy().transpose(1, 2, 0))
    ax[1].set_title('Attacking noise')
    ax[2].imshow(reverse_trans(adv_img[0]))
    attack_class = ', '.join(attack_class.split(',')[:2])
    ax[2].set_title('Adversarial example: {}'.format(attack_class))
    for i in range(3):
        ax[i].set_axis_off()
    plt.tight_layout()

    # plt.show()
    fig.savefig('adv1.png', dpi=fig.dpi)


def graph_result(epsilons, y1, y2, y3, title):
    fig = plt.figure()

    # xnew = np.linspace(min(epsilons), max(epsilons), 20)
    # y1_smooth = spline(epsilons, y1, xnew)
    # l1, = plt.plot(xnew, y1_smooth, 'r--', label="fast")

    l1, = plt.plot(epsilons, y1, 'r--', label="fast")
    l2, = plt.plot(epsilons, y2, 'b--', label="iter non-target")
    l3, = plt.plot(epsilons, y3, 'g--', label="iter target")

    plt.xlabel('epsilon')
    plt.ylabel('%s equivalence' % title)

    plt.legend(handles=[l1, l2, l3])
    # plt.show()

    fig.savefig(title + '.png', dpi=fig.dpi)


def fgsm(img):
    img = img.cpu()
    label = torch.zeros(1, 1).cpu()
    
    x, y = Variable(img, requires_grad=True), Variable(label)
    zero_gradients(x)
    out = model(x)
    y.data = out.data.max(1)[1]
    _loss = loss(out, y)
    _loss.backward()
    # normed_grad = eps * torch.sign(x.grad.data)
    normed_grad = torch.sign(x.grad.data)
    step_adv = x.data + normed_grad
    adv = step_adv - img
    adv = torch.clamp(adv, -eps, eps)
    result = img + adv
    result = torch.clamp(result, 0.0, 1.0)

    return result.cpu(), adv.cpu()


def non_targeted_attack(img):
    img = img.cpu()
    label = torch.zeros(1, 1).cpu()
    
    x, y = Variable(img, requires_grad=True), Variable(label)
    for step in range(steps):
        zero_gradients(x)
        out = model(x)
        y.data = out.data.max(1)[1]
        _loss = loss(out, y)
        _loss.backward()
        normed_grad = step_alpha * torch.sign(x.grad.data)
        step_adv = x.data + normed_grad
        adv = step_adv - img
        adv = torch.clamp(adv, -eps, eps)
        result = img + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result.cpu(), adv.cpu()


def targeted_attack(img, label):
    img = img.cpu()
    label = torch.Tensor([label]).long().cpu()
    
    x, y = Variable(img, requires_grad=True), Variable(label)
    for step in range(steps):
        zero_gradients(x)
        out = model(x)
        _loss = loss(out, y)
        _loss.backward()
        normed_grad = step_alpha * torch.sign(x.grad.data)
        step_adv = x.data - normed_grad
        adv = step_adv - img
        adv = torch.clamp(adv, -eps, eps)
        result = img + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result.cpu(), adv.cpu()

# non-targeted
# img = load_image('img/0248_2.JPG')
# adv_img, noise = non_targeted_attack(img)
# draw_result(img, noise, adv_img)

# targeted
target = randint(0, 999)
img = load_image('img/8.jpg')
adv_img, noise = targeted_attack(img, target)
draw_result(img, noise, adv_img)


# # load images from img dir
# fnames = [f for f in os.listdir('img') if '.jpg' in f]
# images = []
# for f in fnames:
#     img = load_image('img/' + f)
#     images.append((f, img))

# # compare different methods across eps
# epsilons = [0.005, 0.08, 0.16, 0.24, 0.32, 0.4, 0.5]
# n = 20
# methods = ['fgsm', 'non_targ', 'targ']
# y1_1 = []
# y2_1 = []
# y3_1 = []
# y1_5 = []
# y2_5 = []
# y3_5 = []

# for e in epsilons:
#     eps = e

#     for m in methods:
#         tot1 = 0.0
#         tot5 = 0.0

#         for (f, img) in images[:n]:
#             print 'img: %s, eps: %f, method: %s' % (f, eps, m)
#             target = randint(0, 999)

#             # generate adversarial images for all methods
#             if m == 'fgsm':
#                 adv_img, noise = fgsm(img)
#             elif m == 'non_targ':
#                 adv_img, noise = non_targeted_attack(img)
#             else:
#                 adv_img, noise = targeted_attack(img, target)

#             # check if top class is equivalent
#             orig_class, attack_class = get_class(img), get_class(adv_img)
#             if orig_class == attack_class:
#                 tot1 += 1

#             # check if orig class in top 5
#             top5 = get_top_five(adv_img)
#             if orig_class in top5:
#                 tot5 += 1

#         tot1 /= n
#         tot5 /= n

#         if m == 'fgsm':
#             y1_1.append(tot1)
#             y1_5.append(tot5)
#         elif m == 'non_targ':
#             y2_1.append(tot1)
#             y2_5.append(tot5)
#         else:
#             y3_1.append(tot1)
#             y3_5.append(tot5)

# # graph results of each method at each epsilon
# print y1_1, y2_1, y3_1
# graph_result(epsilons, y1_1, y2_1, y3_1, 'top-1')

# print y1_5, y2_5, y3_5
# graph_result(epsilons, y1_5, y2_5, y3_5, 'top-5')

# fgsm
# img = load_image('img/8.jpg')
# adv_img, noise = fgsm(img)
# draw_result(img, noise, adv_img)
