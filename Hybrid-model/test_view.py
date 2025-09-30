import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import skimage.io as io
import torch.nn.functional as F
import torch
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


source_dir = "../../Training7/Result/GeneratedProb/lin1-10-train-200/Hybrid_BN_A/output"
target_dir = "../../Training7/Result/GeneratedLabel/lin1-10-train-200"
if not(os.path.exists(target_dir)):
    os.mkdir(target_dir)

# obtain file list
file_list = generate_file_list(source_dir, 'npy')
for f in file_list:
    filename = get_file_name(f)
    filename_without_ext = f.split('.')[0]
    prediction_prob = np.load(f)
    h,w,d = prediction_prob.shape
    prediction_prob = prediction_prob.reshape((h*w,d))
    prediction_soft_prob = F.softmax(torch.FloatTensor(prediction_prob), dim=1)
    # 0,1 2 3 for other non-ill ill background
    print(prediction_prob.shape)
    prediction_class = np.argmax(prediction_soft_prob, axis=1)
    prediction_class = prediction_class.reshape((h,w))
    print(np.unique(prediction_class))
    # png_img = io.imread(f)
    # print(png_img.shape)
    # print(np.unique(png_img[:,:,0]))
    # print(np.unique(png_img[:,:,1]))
    output_tif = np.zeros((h,w,3),dtype=np.uint8)
    output_tif[prediction_class==1,0] = 255
    output_tif[prediction_class==2,1] = 255
    output_tif[prediction_class==3,2] = 255
    
    prediction_confusion = np.zeros((h*w),dtype=np.uint8)
    prediction_confusion[np.max(prediction_soft_prob.numpy(),axis=1)< 0.6] = 255
    prediction_confusion = prediction_confusion.reshape((h,w))
    im = Image.fromarray(output_tif)
    im.save(os.path.join(target_dir,filename+".tif"))
    im_conf = Image.fromarray(prediction_confusion)
    im_conf.save(os.path.join(target_dir,filename+"_confusion.tif"))
    # print(f, " Done")s