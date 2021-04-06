# Superpixel Sampling Networks
PyTorch implementation of Superpixel Sampling Networks  
paper: https://arxiv.org/abs/1807.10174  
original code: https://github.com/NVlabs/ssn_superpixels

# Requirements
- PyTorch >= 1.4
- scikit-image
- matplotlib

# Usage
## inference
SSN_pix
```
python inference --image /path/to/image
```
SSN_deep
```
python inference --image /path/to/image --weight /path/to/pretrained_weight
```

## training
```
python train.py --root /path/to/BSDS500
```

# Results
SSN_pix  
<img src=https://github.com/perrying/ssn-pytorch/blob/master/SSN_pix_result.png>

SSN_deep  
<img src=https://github.com/perrying/ssn-pytorch/blob/master/SSN_deep_result.png>



# To inference
export CPLUS_INCLUDE_PATH=/usr/local/cuda-10.2/targets/x86_64-linux/include/:$CPLUS_INCLUDE_PATH
CUDA_VISIBLE_DEVICES=0 python inference.py --image /home/janischl/ssn-pytorch/img/2_1_4_cut.png --weight /home/janischl/ssn-pytorch/log/bset_model.pth

# To train 
 CUDA_VISIBLE_DEVICES=0 python train.py --root /home/janischl/ssn-pytorch/data/BSR

# trubleshootiiin
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

device 1
pip install torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html