import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable
#drive.mount('/content/gdrive')
data_dir = '/home/janischl/ssn-pytorch/classify/train'
test_transforms = transforms.Compose([ transforms.Resize((224,300)),
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225])
                                     ])



import math
import numpy as np
import torch
import argparse
import os
import imutils
import cv2
from skimage.color import rgb2lab
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from lib.ssn.ssn import sparse_ssn_iter

def predict_image(image):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model=torch.load('/home/janischl/ssn-pytorch/classify/aerialmodel.pth')
    model.eval()
    model
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index 

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="/path/to/image")
    parser.add_argument("--weight", default=None, type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=50, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=650, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    args = parser.parse_args()




    #image = plt.imread(args.image)
    image = cv2.imread(args.image)
    s = time.time()
    label = inference(image, args.nspix, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight)
   
    #plt.imsave("resultss.png", mark_boundaries(image, label))
    #plt.imsave("avg.png", color.label2rgb(label,image,kind='avg'))
    segment = image.copy()

   
   
    #colors = np.array([[0,0,0]], dtype=np.float32)
    #red = color.label2rgb(label, image, colors=colors, alpha=0.3, bg_label=3, bg_color=(1, 0, 0), image_alpha=1, kind='overlay')
    #red[label!=3] = 0
    #plt.imsave("plsegment.png",segment)
    #regions = measure.regionprops(label, intensity_image=image)
    #print([r.area for r in regions])
    #print([r.mean_intensity for r in regions])
    ####

    mask=image.copy()
    for i in range(2,500):
        i=i+1
        print(i)
        segment = image.copy()
        segment[label!=i] = 0
        
        gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
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
               
                crop_img = segment[topmost[1]:bottommost[1],leftmost[0]:rightmost[0]]
            

    #           cv2.imwrite(('superpixel/'+ str(i)+'crop.png'),crop_img)



          

            model=torch.load('/home/janischl/ssn-pytorch/classify/aerialmodel.pth')
            model.eval()
            model


            to_pil = transforms.ToPILImage()
            #images, labels = get_random_images(5)
            #fig=plt.figure(figsize=(10,10))
            imagee = to_pil(crop_img)
            index = predict_image(imagee)
            print(index)
            if (index == 0):
                mask[label==i] = [255,255,255]
            cv2.imwrite(('/home/janischl/ssn-pytorch/classify/mask.png'),mask)
            #sub = fig.add_subplot(1, len(crop_img), ii+1)
            #res = int(labels[ii]) == index
            #sub.set_title(str([index]) + ":" + str(res))
            #print(res)
                #plt.axis('off')
                #plt.imshow(image)
            #plt.show()




                
           

     
