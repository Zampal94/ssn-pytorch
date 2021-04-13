import sys
sys.path.insert(1, '/home/janischl/ssn-pytorch')
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

import time
import argparse
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage import color

import math
import argparse
import os
import imutils
import cv2
from skimage.color import rgb2lab
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from lib.ssn.ssn import sparse_ssn_iter


@torch.no_grad()
def inference(image, nspix, n_iter, fdim=None, color_scale=0.26, pos_scale=2.5, weight=None, enforce_connectivity=True):

    if weight is not None:
        from model import SSNModel
        model = SSNModel(fdim, nspix, n_iter).to("cuda")
        checkpoint = torch.load(weight)
        model.load_state_dict(checkpoint)#['model_state_dict']
        model.eval()
    else:
        model = lambda data: sparse_ssn_iter(data, nspix, n_iter)

    height, width = image.shape[:2]

    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)    

    coords = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda")), 0)
    coords = coords[None].float()

    image = rgb2lab(image)
    image = torch.from_numpy(image).permute(2, 0, 1)[None].to("cuda").float()

    inputs = torch.cat([color_scale*image, pos_scale*coords], 1)

    _, H, _ = model(inputs)

    labels = H.reshape(height, width).to("cpu").detach().numpy()

    if enforce_connectivity:
        segment_size = height * width / nspix
        min_size = int(0.02 * segment_size)   #0.06
        max_size = int(3 * segment_size)    #3
        labels = _enforce_label_connectivity_cython(
            labels[None], min_size, max_size)[0]

    return labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image",default="/home/janischl/ssn-pytorch/img/1_1_1.png", type=str, help="/path/to/image")
    parser.add_argument("--weight", default="/home/janischl/ssn-pytorch/log/bset_model.pth", type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=50, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=650, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    args = parser.parse_args()


    image = cv2.imread('/home/janischl/ssn-pytorch/img_luca/8_7_4.png') 
    mask = cv2.imread('/home/janischl/ssn-pytorch/classify/mask_generated_8_7_4_onlyref_3class.png') 
    s = time.time()
    
    label = inference(image, args.nspix, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight)
    
    segment = image.copy()
    mask=mask.copy()
    image_vec = []
    for i in range(0,1500):
    
        print(i)
        segment = image.copy()
        segment[label!=i] = 0
        
        gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        oldhorizontal = 0
        oldvertikal = 0
        lst_intensities = []
        b=-1
        for c in cnts:
            b+=1
            leftmost = tuple(c[c[:,:,0].argmin()][0])
            rightmost = tuple(c[c[:,:,0].argmax()][0])
            topmost = tuple(c[c[:,:,1].argmin()][0])
            bottommost = tuple(c[c[:,:,1].argmax()][0]) 
            if (topmost[1]!=bottommost[1] and leftmost[0]!=rightmost[0]): 
                #mask[lst_intensities]=[2,2,2]
                print(bottommost) 
                print(leftmost) 
                print(rightmost) 
                print(topmost) 
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                r,g,b = mask[cY, cX]
                print(r,g,b)
                
                cimg= np.zeros_like(image.copy())
                cv2.drawContours(cimg, cnts, b, color=255, thickness=-1)
                pts = np.where(cimg == 255)
                #print(pts[0])
                
                mask[pts[0],pts[1]] = [55,55,55]
                #lst_intensities.append(segment[pts[0], pts[1]])
                #print(lst_intensities)

                r,g,b = mask[cY, cX]
                print(r,g,b)
                
                horizontal = rightmost[0]- leftmost[0] 
                vertikal = bottommost[1]-topmost[1]
                if (horizontal > oldhorizontal or vertikal > oldvertikal):
                    oldhorizontal = horizontal
                    oldvertikal = vertikal

                    crop_mask = mask[topmost[1]:bottommost[1],leftmost[0]:rightmost[0]]
                   