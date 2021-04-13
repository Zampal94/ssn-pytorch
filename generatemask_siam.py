import sys
sys.path.insert(1, '/home/janischl/deep-metric-learning-tsinghua-dogs/src/')
from infer_siamese import main
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable

data_dir = '/home/janischl/ssn-pytorch/classify/train'

import math
import argparse
import os
import imutils
import cv2
from skimage.color import rgb2lab
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from lib.ssn.ssn import sparse_ssn_iter

def predict_image(image_vec):
    output, path = main(image_vec)
    return output, path 

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
    import time
    import argparse
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    from skimage import color
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",default="/home/janischl/ssn-pytorch/img/1_1_1.png", type=str, help="/path/to/image")
    parser.add_argument("--weight", default="/home/janischl/ssn-pytorch/log/bset_model.pth", type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=50, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=650, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    args = parser.parse_args()


    image = cv2.imread('/home/janischl/ssn-pytorch/img_luca/2_7_4.png') 
    s = time.time()
    label = inference(image, args.nspix, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight)

    segment = image.copy()
    mask=image.copy()
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
        for c in cnts:
            M = cv2.moments(c)
         
            leftmost = tuple(c[c[:,:,0].argmin()][0])
            rightmost = tuple(c[c[:,:,0].argmax()][0])
            topmost = tuple(c[c[:,:,1].argmin()][0])
            bottommost = tuple(c[c[:,:,1].argmax()][0]) 
            if (topmost[1]!=bottommost[1] and leftmost[0]!=rightmost[0]): 
                print(bottommost) 
                print(leftmost) 
                print(rightmost) 
                print(topmost) 
               
                horizontal = rightmost[0]- leftmost[0] 
                vertikal = bottommost[1]-topmost[1]
                
                if (horizontal > oldhorizontal or vertikal > oldvertikal):
                    oldhorizontal = horizontal
                    oldvertikal = vertikal

                    if (vertikal<224):
                        restv = 224 - vertikal
                    else:
                        restv=0
                    if (horizontal<224):
                        resth = 224 - horizontal
                    else:
                        resth = 0
                    
                    cleftmost = leftmost[0] - int(resth/2)
                    crightmost = rightmost[0] + int(resth/2)
                    cbottommost = bottommost[1] + int(restv/2)
                    ctopmost = topmost[1] - int(restv/2)

                    if (cleftmost>=1 and crightmost<=2391 and ctopmost>=1 and cbottommost<=1143):        
                        crop_img = segment[ctopmost:cbottommost,cleftmost:crightmost]
                    else:
                        if (vertikal<224):
                            restv = 224 - vertikal
                        else:
                            restv=0
                        if (horizontal<224):
                            resth = 224 - horizontal
                        else:
                            resth = 0
                        crop_img = segment[topmost[1]:bottommost[1],leftmost[0]:rightmost[0]] 
                        crop_img = cv2.copyMakeBorder(crop_img, int(restv/2), int(restv/2), int(resth/2), int(resth/2), cv2.BORDER_CONSTANT, value=[0, 0, 0])         
                        #crop_img = np.pad(crop_img[:,:,2], ((int(restv/2),int(restv/2)),(int(resth/2),int(resth/2))), mode='constant', constant_values=0)
                        
                        cv2.imwrite("/home/janischl/ssn-pytorch/classify/added.png",crop_img)
                    
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        to_pil = transforms.ToPILImage()
    
        imagee = transforms.ToPILImage()(crop_img)
        image_vec.append(imagee)
        #print(image_vec)
    distance, paths = predict_image(image_vec)

     
    print("distance")
    print(distance)
    k = 0
    for path in paths:
        head, tail = os.path.split(path)
        print(head)
        background_path = "/home/janischl/Dataset/imagenet/images/valid/Background"

        tool_path = "/home/janischl/Dataset/imagenet/images/valid/Tool"
     
        flankwear_path = "/home/janischl/Dataset/imagenet/images/valid/Flankwear"

        if(head == background_path):
            print("background")
            mask[label==k] = [0,0,0]
    
            
        if(head == tool_path):
            print("tool")
            mask[label==k] = [5,5,225] 
     
        if (head == flankwear_path):
            mask[label==k] = [255,255,255]
    
        cv2.imwrite(('/home/janischl/ssn-pytorch/classify/mask_generated__onlyref_3class.png'),mask)
        k=k+1

                     # if (index == 0):
        #     mask[label==i] = [0,0,0]
        # if (index==3):
        #     mask[label==i] = [0,240,255]
        # if (index==4):
        #     mask[label==i] = [200,200,200]
        # if (index==5):
        #     mask[label==i] = [200,200,200]
        # if (index==3):
        #     mask[label==i] = [5,5,225]  
           

     
