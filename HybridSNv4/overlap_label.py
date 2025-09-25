import os
from pathlib import Path
import sys
import imageio.v2 as iio
import numpy as np
from PIL import Image

def generate_file_list(dir, end):
    list = [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith(end)]
    list.sort()
    return list

def get_file_name(path):
    return Path(path).stem

# read and process label tif image, only keep red channel and change[0, 255] to [0, 1]
def read_tif_img(file):
    return np.array(iio.imread(file), dtype = np.uint8)

def process_tif_img(img):
    if(len(img.shape)>2):
        r_img = img[:,:,0]
        r_img[r_img>250]=1
        return r_img
    else:
        img[img>250]=1
        return img

def read_process_tif_img(file):
    return process_tif_img(read_tif_img(file))


ill_label_dir = "../RawData/betterNorm/ill"
cell_label_dir = "../RawData/betterNorm/cell"
target_dir = "../RawData/betterNorm/label"
# obtain file list
file_list = generate_file_list(ill_label_dir, 'tif')

# make sure number of files to be processed is correct
modifiableNum = 0
for f in file_list:
    filename = get_file_name(f)
    if os.path.exists(os.path.join(cell_label_dir,filename+".tif")):
        modifiableNum += 1

if not(os.path.exists(target_dir)):
    os.mkdir(target_dir)

# Processing files, 2 for ill cell, 1 for not ill cell, 0 for other
if int(input("Number of files pair to be overlapped is " + str(modifiableNum) + " if correct enter 1\n")) == 1:
    for f in file_list:
        filename = get_file_name(f)
        print(filename)
        # Value 0 1 2
        ill_label = read_process_tif_img(os.path.join(ill_label_dir,filename+".tif"))
        cell_label = read_process_tif_img(os.path.join(cell_label_dir,filename+".tif"))
        print(np.unique(ill_label))
        print(np.unique(cell_label))
        ill_cell_label = np.where(cell_label == 1, cell_label+ill_label, cell_label)
        print("File " + filename + " has processed unique value " + str(np.unique(ill_cell_label)))
        # Graphical, channel overlap
        graph_ill_cell_label =  np.zeros((ill_label.shape[0],ill_label.shape[1],3), dtype=np.uint8)
        graph_ill_cell_label[:,:,0] = np.where(ill_cell_label==1,255,0)
        graph_ill_cell_label[:,:,1] = np.where(ill_cell_label==2,255,0)

        im = Image.fromarray(ill_cell_label)
        im.save(target_dir+"/"+filename + ".tif")
        graphim = Image.fromarray(graph_ill_cell_label)
        graphim.save(target_dir+"/"+filename + ".png")
