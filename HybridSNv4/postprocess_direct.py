import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import skimage.io as io
import torch.nn.functional as F
import torch

import cv2
import matplotlib.pyplot as plt
from tqdm import *

def generate_file_list(dir, end):
    list = [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith(end)]
    list.sort()
    return list

def get_file_name(path):
    return Path(path).stem

# read and process label tif image, only keep red channel and change[0, 255] to [0, 1]
def read_tif_img(file):
    return io.imread(file)

def process_tif_img(img):
    if(len(img.shape)>2):
        r_img = img[:,:,0]
        r_img[r_img==255]=1
        return r_img
    else:
        return img

def softmax(x):
    max = np.max(x,axis=1,keepdims=True)
    e_x = np.exp(x-max)
    sum = np.sum(e_x,axis=1,keepdims=True)
    return e_x/sum

def read_process_tif_img(file):
    return process_tif_img(read_tif_img(file))


source_dir = "../../Training7/Result/GeneratedProb/1-8/RBF_SVM/output"
target_dir = "../../Training7/Result/GeneratedLabel/1-8/RBF_SVM"

#source_dir = "D:/Training7/GeneratedProb/lin1-10/Hybrid_BN_A/output"
#target_dir = "D:/Training7/PostProcess/lin1-10"
polished_dir = target_dir+"/polished_output"
noisy_dir = target_dir+"/noisy_output"
each_type_dir = target_dir+"/each_type"
soft_prob_thresh = 0.5
connected_thresh = 50
connectivityParam = 8
net = 0

class_name = ["non-cell","non-ill-cell","ill-cell","background"]
if not(os.path.exists(target_dir)):
    os.mkdir(target_dir)
if not(os.path.exists(polished_dir)):
    os.mkdir(polished_dir)
if not(os.path.exists(noisy_dir)):
    os.mkdir(noisy_dir)
if not(os.path.exists(each_type_dir)):
    os.mkdir(each_type_dir)
# obtain file list
file_list = generate_file_list(source_dir, 'npy')
for f in file_list:
    filename = get_file_name(f)
    filename_without_ext = f.split('.')[0]
    prediction_prob = np.load(f)
    h,w,d = prediction_prob.shape
    prediction_prob = prediction_prob.reshape((h*w,d))
    if net:
        prediction_soft_prob = F.softmax(torch.FloatTensor(prediction_prob), dim=1)
        prediction_soft_prob = prediction_soft_prob.numpy()
    else:
        prediction_soft_prob = prediction_prob
    filtered_class = np.zeros((h,w,4),dtype=np.uint8)

    # 0,1 2 3 for other non-ill ill background

    prediction_class = np.argmax(prediction_soft_prob, axis=1)
    prediction_class = prediction_class.reshape((h,w))
    print(np.unique(prediction_class))

    output_tif = np.zeros((h,w,3),dtype=np.uint8)
    output_tif[prediction_class == 1] = [255,0,0]
    output_tif[prediction_class == 2] = [0,255,0]
    output_tif[prediction_class == 3] = [0,0,255]
    output_tif[prediction_class == 4] = [255,255,255]
    im = Image.fromarray(output_tif)
    im.save(os.path.join(noisy_dir,filename+".tif"))
    print(f, " Done")
