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


source_dir = "../../Training7/Result/GeneratedProb/1-8/RF/output"
target_dir = "../../Training7/Result/GeneratedLabel/1-8/RF"

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
    else:
        prediction_soft_prob = prediction_prob
    filtered_class = np.zeros((h,w,4),dtype=np.uint8)
    # For each class
    for type in [0,1,2,3]:
        type_name = class_name[type]
        # generate heat map
        data = prediction_soft_prob[:,type]
        if net:
            data = data.numpy()
        data = data.reshape((h,w))
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w/100,h/100)
        ax = plt.Axes(fig, [0.,0.,1.,1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(data, cmap='hot', interpolation='nearest',aspect = 'auto')
        fig.savefig(os.path.join(each_type_dir,filename+"_"+type_name+".tif"))
        # generate filtered components
        ret, thresh = cv2.threshold(data,soft_prob_thresh,1,cv2.THRESH_BINARY)
        thresh_uint = thresh.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_uint, connectivity = connectivityParam)
        areas = list()
        for i in range(num_labels):
            areas.append(stats[i][-1])
        image_filtered = np.zeros_like(thresh_uint)
        for (i,label) in enumerate(np.unique(labels)):
            if label == 0:
                continue
            if stats[i][-1] > connected_thresh:
                image_filtered[labels == i] = 255
        filtered_class[:,:,type] = image_filtered
        im = Image.fromarray(image_filtered)
        im.save(os.path.join(each_type_dir,filename+"_"+type_name+"_filtered.tif"))
        # plt.show()
        plt.close()
    
    # filtered_class done, check sum to decide pixel
    filtered_class[filtered_class==255] = 1
    filtered_class_sum = np.sum(filtered_class,axis = 2)        
    # 0,1 2 3 for other non-ill ill background

    # print(prediction_prob.shape)
    # prediction_class = np.argmax(prediction_soft_prob, axis=1)
    # prediction_class = prediction_class.reshape((h,w))
    # print(np.unique(prediction_class))
    # png_img = io.imread(f)
    # print(png_img.shape)
    # print(np.unique(png_img[:,:,0]))
    # print(np.unique(png_img[:,:,1]))
    output_type = np.zeros((h,w),dtype=np.uint8)
    output_type[filtered_class[:,:,0]==1] = 0
    output_type[filtered_class[:,:,1]==1] = 1
    output_type[filtered_class[:,:,2]==1] = 2
    output_type[filtered_class[:,:,3]==1] = 3
    output_type[filtered_class_sum == 0] = 4


    output_tif = np.zeros((h,w,3),dtype=np.uint8)
    output_tif[output_type == 1] = [255,0,0]
    output_tif[output_type == 2] = [0,255,0]
    output_tif[output_type == 3] = [0,0,255]
    output_tif[output_type == 4] = [255,255,255]
    im = Image.fromarray(output_tif)
    im.save(os.path.join(noisy_dir,filename+".tif"))
    
    
    # filter undefined pixel, iterate all undefined pixel connected components, find surrounding pixel type, 
    # change it based on non-ill-cell > ill-cell > background > non-cell order, or 1>2>3>0
    undefined_pixel = np.where(output_type==4,1,0)
    undefined_pixel = undefined_pixel.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(undefined_pixel, connectivity = connectivityParam)
    print("number of undefined connected components is ", num_labels)
    for (i,label) in tqdm(enumerate(np.unique(labels))):
        if label == 0:
            continue
        label_mask = np.zeros((h,w),np.uint8)
        label_mask[labels==i] = 1
        kernel = np.ones((3,3),np.uint8)
        dilated_label_mask = cv2.dilate(label_mask, kernel, iterations = 1)
        neighbors = dilated_label_mask - label_mask
        # print(np.unique(output_type[dilated_label_mask==1]))
        # print(np.unique(output_type[label_mask==1]))
        neighbor_types = np.unique(output_type[neighbors == 1])
        neighbor_types = np.delete(neighbor_types, neighbor_types == 4)
        neighbor_types = np.sort(neighbor_types)
        # Since neighbor types are sorted, 1230 priority, for multiple elements, choose the first one which is not zero. For other, pick the first element
        # print(neighbor_types)
        # print("area size", stats[i][-1])
        if neighbor_types.shape[0] > 1 and 0 in neighbor_types:
            output_type[label_mask==1] = neighbor_types[1]
        else:
            output_type[label_mask==1] = neighbor_types[0]
    
    
    # Change 1 2 label.
    cell_pixel = np.where(output_type==1,1,0) + np.where(output_type == 2,1,0)
    cell_pixel = cell_pixel.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cell_pixel, connectivity = connectivityParam)
    print("number of cell connected components is ", num_labels)
    for (i,label) in tqdm(enumerate(np.unique(labels))):
        if label == 0:
            continue
        label_mask = np.zeros((h,w),np.uint8)
        label_mask[labels==i] = 1
        cell_value = output_type[label_mask==1]
        ill_num = np.count_nonzero(cell_value==2)
        non_ill_num = np.count_nonzero(cell_value==1)
        # Now using majority vote
        if (ill_num >= non_ill_num):
            output_type[label_mask == 1] = 2
        else:
            output_type[label_mask == 1] = 1
            
    # generate output tif
    output_tif = np.zeros((h,w,3),dtype=np.uint8)
    output_tif[output_type == 1] = [255,0,0]
    output_tif[output_type == 2] = [0,255,0]
    output_tif[output_type == 3] = [0,0,255]
    # output_tif[output_type == 4] = [255,255,255]
    
    # prediction_confusion = np.zeros((h*w),dtype=np.uint8)
    # prediction_confusion[np.max(prediction_soft_prob.numpy(),axis=1)< 0.6] = 255
    # prediction_confusion = prediction_confusion.reshape((h,w))
    
    im = Image.fromarray(output_tif)
    im.save(os.path.join(polished_dir,filename+".tif"))
    
    
    # im_conf = Image.fromarray(prediction_confusion)
    # im_conf.save(os.path.join(target_dir,filename+"_confusion.tif"))
    plt.close()
    print(f, " Done")
