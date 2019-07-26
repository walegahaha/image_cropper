from __future__ import division
import os
import shutil
import copy
from tqdm import tqdm
import time
import numpy as np
from PIL import Image, ImageDraw

def MaxSubarrayFL(array, w, T):
    for i in range(1, len(array)): array[i] += array[i-1]
    j1 = -1
    Smax = -1
    wr = -1
    w1, w2 = min(w), max(w)
    if T > 0 and w1 > 0:
        for st in range(1, len(array)-w2+1):
            for offset in range(w2-w1+1):
                S0 = array[st+offset+w1-1] - (array[st-1] if st>=1 else 0)
                if S0 >= T and S0 > Smax:
                    j1 = st 
                    wr = w1 + offset
                    Smax = S0
    return j1, wr, Smax

def FixedAspRatioRectangle(G, alpha, ratio):
    m, n = G.shape
    GcSum = copy.deepcopy(G)
    for i in range(1, len(G)):
        GcSum[i] += GcSum[i-1]
    GSum = copy.deepcopy(GcSum)
    for i in range(1, len(G[0])):
        GSum[:,i] += GSum[:,i-1]
    i, j, w, h = 0, 0, n, m
    i1, i2, T, Smin = 0, 0, alpha*GSum[m-1,n-1], -1

    if GSum[m-1,n-1]/(255*m*n) < 0.1:
        return i, j, w, h, 0.10
    
    while i1 < m-1 and i2 < m:
        h0 = i2 - i1 + 1
        w0 = int(h0 / ratio)
        if w0 > n:
            i1 += 1
        else:
            array = GcSum[i2, :] - (GcSum[i1-1, :] if i1 >=1 else 0)
            j1, wr, S0 = MaxSubarrayFL(array, (h0, w0), T)
            if j1 > -1:
                if wr*h0 < w*h or (wr*h0 == w*h and S0 < Smin):
                    i, j, w, h = i1, j1, wr, h0
                    Smin = S0 
                i1 += 1
            else:
                i2 += 1
    return i, j, w, h, Smin if Smin == -1 else Smin/GSum[m-1,n-1]


if os.path.exists('./crops'):
    shutil.rmtree('./crops')
os.mkdir('./crops')

filenames = sorted(os.listdir('./masks'))
for i in tqdm(range(len(filenames))):
    filename = filenames[i]
    image = Image.open('../test_images/{:s}'.format(filename))
    ori_shape = image.size
    hw_ratio = ori_shape[1] / ori_shape[0]
    if ori_shape[0] < ori_shape[1]:
        mask_shape = (224, int(224*hw_ratio))
    else:
        mask_shape = (int(224/hw_ratio), 224)
    
    mask = Image.open('./masks/{:s}'.format(filename)).resize(mask_shape, Image.ANTIALIAS)
    npMask = np.array(mask.convert('L')).astype(np.float32)

    #npMask[npMask < 100] = 0
    #npMask[npMask > 100] = 255

    i, j, w, h, trueAlpha = FixedAspRatioRectangle(npMask, 0.9, 4/3)
    w_resize_ratio, h_resize_ratio = np.array(ori_shape) / np.array(mask_shape)
    i, j, w, h = i*h_resize_ratio, j*w_resize_ratio, w*w_resize_ratio, h*h_resize_ratio
    
    draw = ImageDraw.Draw(image)
    draw.rectangle((j, i, j+w, i+h), outline=(255,0,0), width=2)
    image.save('crops/{:s}_{:.2f}.jpg'.format(filename[:-4], trueAlpha))


