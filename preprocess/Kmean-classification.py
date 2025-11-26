import numpy as np
import os
from spectral import *
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm
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

# ----------------------------------------------------------------------CONFIGURATIONS--------------------------------------------------------------
KMEAN_NUM = 20
input_dir = "../processed_dataset/datacubes"
sam_thresh = 0.6
# ----------------------------------------------------------------------CONFIGURATIONS ENDS------------------------------------------------


# output_dir = "x-sg-x-" + str(KMEAN_NUM)
output_dir = input_dir
if not(os.path.exists(output_dir)):
    os.mkdir(output_dir)



def SAM(x,y):
    s = np.sum(np.dot(x,y))
    t = np.sqrt(np.sum(x**2))*np.sqrt(np.sum(y**2))
    th = np.arccos(s/t)
    return th
fileList = generate_file_list(input_dir, "hdr")

totalFileNumber = len(fileList)
finishedFileNumber = 0
for input_file in tqdm(fileList):
    print("Now processing ", input_file)
    data = read_hdr_file(input_file)
    h,w,b = data.shape
    data = data.reshape((data.shape[0]*data.shape[1],data.shape[2]))
    kmeans = KMeans(n_clusters=KMEAN_NUM, random_state=0, n_init="auto").fit(data)
    label = kmeans.labels_
    label = label.reshape((h,w))
    output_file = os.path.join(output_dir,get_base_name(input_file)+".npy")
    print("Save output file ", output_file)
    np.save(output_file, label)
    label_num = np.unique(label).shape[0]
    print(f"unique output labels are {np.unique(label)}")
    colors = np.random.rand(label_num*3)*255
    colors = colors.astype(np.uint8)
    colors = colors.reshape((label_num,3))
    img = colors[label]
    cv2.imwrite(output_dir + "/" + get_base_name(output_file) + ".tiff", img)
    print("Save output image ", output_dir + "/" + get_base_name(output_file) + ".tiff" )
    finishedFileNumber += 1
