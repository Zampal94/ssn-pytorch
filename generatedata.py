
import sys

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
import numpy as np
import argparse
import os
import imutils
import cv2
from skimage.color import rgb2lab
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from lib.ssn.ssn import sparse_ssn_iter


@torch.no_grad()
def inference(image, nspix, n_iter, fdim=None, color_scale=0.26, pos_scale=2.5, weight=None, enforce_connectivity=True):
    """
    generate superpixels

    Args:
        image: numpy.ndarray
            An array of shape (h, w, c)
        nspix: int
            number of superpixels
        n_iter: int
            number of iterations
        fdim (optional): int
            feature dimension for supervised setting
        color_scale: float
            color channel factor
        pos_scale: float
            pixel coordinate factor
        weight: state_dict
            pretrained weight
        enforce_connectivity: bool
            if True, enforce superpixel connectivity in postprocessing

    Return:
        labels: numpy.ndarray
            An array of shape (h, w)
    """
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
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",default="/home/janischl/ssn-pytorch/img/1_1_1.png", type=str, help="/path/to/image")
    parser.add_argument("--weight", default="/home/janischl/ssn-pytorch/log/bset_model.pth", type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=50, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=650, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    args = parser.parse_args()



    for img in os.listdir("/home/janischl/ssn-pytorch/img"):
        pathname = os.path.basename(img)
        
        filename = os.path.splitext(pathname)[0]
        print(pathname)
        img_path = ("/home/janischl/ssn-pytorch/img/"+pathname)
        print(img_path)
        mask_path = ("/home/janischl/ssn-pytorch/mask/"+filename+".png")
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
  
        
        s = time.time()
        label = inference(image, args.nspix, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight)
    
        segment = image.copy()

    
    
       

        for i in range(0,len(label)):
            
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
                if (topmost[1]<=(bottommost[1]-10) and leftmost[0]<=(rightmost[0]-10)): 
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    r,g,b = mask[cY, cX]
                    print (r,g,b)
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
                            
                            if (r==0 and g==0 and b==0):
                                cv2.imwrite(('/home/janischl/ssn-pytorch/train_0904/Background/'+filename+str(i)+'edge.png'),crop_img)
                            if (r==5 and g==5 and b==255):
                                cv2.imwrite(('/home/janischl/ssn-pytorch/train_0904/Tool/'+filename+str(i)+'edge.png'),crop_img)
                            if (r==0 and g==240 and b==255):
                                cv2.imwrite(('/home/janischl/ssn-pytorch/train_0904/Groove/'+filename+str(i)+'.png'),crop_img)
                            if (r==255 and g==255 and b==255):
                                cv2.imwrite(('/home/janischl/ssn-pytorch/train_0904/Flankwear/'+filename+str(i)+'.png'),crop_img)
                            #if (r==192 and g==192 and b==192):
                            #    cv2.imwrite(('/home/janischl/ssn-pytorch/train_img/Breakage/'+filename+str(i)+'.png'),crop_img)
                            #if (r==0 and g==64 and b==0):
                            #    cv2.imwrite(('/home/janischl/ssn-pytorch/train_img/Groove/'+filename+str(i)+'.png'),crop_img)

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
                            #cv2.imwrite("/home/janischl/ssn-pytorch/classify/noadded.png",crop_img)                  
                            crop_img = cv2.copyMakeBorder(crop_img, int(restv/2), int(restv/2), int(resth/2), int(resth/2), cv2.BORDER_CONSTANT, value=[0, 0, 0])
                            #crop_img = np.pad(crop_img, ((int(restv/2),int(restv/2)),(int(resth/2),int(resth/2)), mode='constant', constant_values=0)
                            #cv2.imwrite("/home/janischl/ssn-pytorch/classify/added.png",crop_img) 

                        
                            if (topmost[1]==0 or bottommost[1]==2390 or leftmost[0]== 0 or rightmost[0]>=1130):
                                if (r==0 and g==0 and b==0):
                                    cv2.imwrite(('/home/janischl/ssn-pytorch/train_0904/Background/'+filename+str(i)+'edge.png'),crop_img)
                                if (r==5 and g==5 and b==255):
                                    cv2.imwrite(('/home/janischl/ssn-pytorch/train_0904/Tool/'+filename+str(i)+'edge.png'),crop_img)
                            #if (r==0 and g==240 and b==255):
                            #    cv2.imwrite(('/home/janischl/ssn-pytorch/train_edge/Groove/'+filename+str(i)+'.png'),crop_img)
                            #if (r==255 and g==255 and b==255):
                            #    cv2.imwrite(('/home/janischl/ssn-pytorch/train_edge/Flankwear/'+filename+str(i)+'.png'),crop_img)
                        # if (r==192 and g==192 and b==192):
                        #     cv2.imwrite(('/home/janischl/ssn-pytorch/train_img/Breakage/'+filename+str(i)+'.png'),crop_img)
                        #if (r==0 and g==64 and b==0):
                        #    cv2.imwrite(('/home/janischl/ssn-pytorch/train_img/Groove/'+filename+str(i)+'.png'),crop_img)
            i=i+1            
                      



                
           

     
