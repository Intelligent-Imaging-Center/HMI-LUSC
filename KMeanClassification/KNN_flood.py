import numpy as np
import math
import os
import copy
# model related
import scipy.signal
from sklearn.decomposition import PCA
# IO related
from spectral import *
import imageio.v2 as iio
import joblib 
import logging
from PIL import Image
import spectral as spy
import cv2
from sklearn.cluster import KMeans

# -------------------------------------Read image and model--------------------------------------------
def generate_file_list(dir, end):
    list = [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith(end)]
    list.sort()
    return list

def get_base_name(path):
    return os.path.basename(path).split(".")[0]
    
    
# Change HDR file into an image array with 69 bands 
def read_hdr_file(file):
    # return np.array(scipy.signal.savgol_filter(open_image(file).load(),5,2))
    return np.array(open_image(file).load())
    # return np.array(open_image(file).load())[:,:,30:31]


# ----------------------------------------------------------------------设置--------------------------------------------------------------
KMEAN_NUM = 20
input_dir = "D:/DataSource/datacube-lin5/snv-sg-snv"
output_dir = "D:/KMeanTest/snv-sg-snv-" + str(KMEAN_NUM)
if not(os.path.exists(output_dir)):
    os.mkdir(output_dir)
sam_thresh = 0.6


def SAM(x,y):
    s = np.sum(np.dot(x,y))
    t = np.sqrt(np.sum(x**2))*np.sqrt(np.sum(y**2))
    th = np.arccos(s/t)
    # print(s,t)
    return th


class hsi_type:
    def __init__(self, typical, name):
        self.typical = typical
        self.name = name
        color = np.random.rand(3)*255
        self.color = color.astype(np.int32)
        self.count = 1
        
    # --------------------------------------------------------------------调整条件-----------------------------------------------------------------
    def checkInclusion(self, target):
        sam_angle = SAM(self.typical, target)
        # print(sam_angle)
        return sam_angle < sam_thresh
        # diff = np.abs(self.typical - target)
        # return np.count_nonzero(diff >= 20) <= 30 and np.count_nonzero(diff >= 35) <= 10  and np.count_nonzero(diff >= 50) <= 0 
        # return np.count_nonzero(diff >= per_band_value_tolerance) <= per_band_number_tolerance and np.sum(diff) <= total_band_value_tolerance
        
    def rename(self, new_name):
        self.name = new_name

    def inc_count(self):
        self.count += 1
fileList = generate_file_list(input_dir, "hdr")

totalFileNumber = len(fileList)
finishedFileNumber = 0
for input_file in fileList:
    print("Now processing ", input_file)
    data = read_hdr_file(input_file)

    print(data.shape)
    h,w,b = data.shape
    data = data.reshape((data.shape[0]*data.shape[1],data.shape[2]))
    print(data.shape)
    kmeans = KMeans(n_clusters=KMEAN_NUM, random_state=0, n_init="auto").fit(data)
    label = kmeans.labels_
    label = label.reshape((h,w))
    print(type(label))
    # data = data[:,:,[45,21,7]]
    
    # height = data.shape[0]
    # width = data.shape[1]
    # band = data.shape[2]
    # label = np.full((height,width),-1, dtype = int) # 0 means the type is not decided yet, must assign a value larger than 0 at the end
    # existing_type_num = -1
    # existing_types = {}
    # # Filling type starts
    # for i in range(0, height):
    #     for j in range(0,width):
    #         if label[i][j] != -1:
    #             continue
    #         else:
    #             pixel_hs = data[i,j,:]
    #             fit_found = False
    #             # check whether this pixel belong to a existing type
    #             for k in range(0, existing_type_num+1):
    #                 if fit_found == False and existing_types[k].checkInclusion(pixel_hs):

    #                     existing_types[k].inc_count()
    #                     label[i,j] = k
    #                     fit_found = True
    #             # if no fit existing, consider this one as a new type
    #             if fit_found == False:
    #                 if(existing_type_num>= 0):
    #                     diff =  np.abs(existing_types[0].typical - pixel_hs)
    #                 existing_type_num += 1
    #                 label[i,j] = existing_type_num
    #                 existing_types[existing_type_num] = hsi_type(pixel_hs, str(existing_type_num))
    #                 print("New type found, now have ", existing_type_num, " types")
    # print(existing_type_num)
    output_file = os.path.join(output_dir,get_base_name(input_file)+".npy")
    print("Save output file ", output_file)
    np.save(output_file, label)
    label_num = np.unique(label).shape[0]
    print(np.unique(label))
    colors = np.random.rand(label_num*3)*255
    colors = colors.astype(np.uint8)
    colors = colors.reshape((label_num,3))
    img = colors[label]

    # print(img.max())
    # print(img.min())
    # print(img.shape)
    # # im = Image.fromarray(img, mode = 'RGB')
    # # im.show()
    # # im.save("my_label/lin1-1.tiff")
    cv2.imwrite(output_dir + "/" + get_base_name(output_file) + ".tiff", img)
    print("Save output image ", output_dir + "/" + get_base_name(output_file) + ".tiff" )
    finishedFileNumber += 1
    print("Finished files ", finishedFileNumber, " / ", totalFileNumber)