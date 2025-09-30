from utils import *
import yaml
import os
import logging
from models.HybridSN import *
from PIL import Image
import spectral as spy
import matplotlib.pyplot as plt
# ------------------------------------------Logging Function-----------------------------------------
if not(os.path.exists("logs")):
    os.mkdir("logs")
if os.path.isfile("logs/load_test.log"):
    os.remove("logs/load_test.log")
logging.basicConfig(filename="logs/load_test.log", format='%(asctime)s %(levelname)-8s %(message)s', 
                    level = logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# define input and target directories
with open('./configs/comparison.yml', 'r') as stream:
    configs = yaml.safe_load(stream)

# ------------------------------------------Read Configuration----------------------------------------

# 输出文件夹，保存(patch, label)对
input_dir = configs["input_dir"]
# Y文件夹
label_dir = configs["label_dir"]
# 输出文件夹，保存(patch, label)对
prediction_dir = configs["prediction_dir"]
test_num = configs["test_num"]
save_dir = configs["save_dir"]
if not(os.path.exists(save_dir)):
    os.mkdir(save_dir)
# -------------------------------------------Read Data and Label-----------------------------------------
for i in range(1, test_num+1):
    ID = str(i)
    X = read_hdr_file(input_dir+"/"+ID+".hdr")
    Y = read_tif_img(label_dir+"/"+ID+".tif")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure(figsize=(10,10))
    rows = 2
    columns = 3
    fig.add_subplot(rows,columns,1)
    spy.imshow(X,stretch=0.02,title="Original",fignum=fig.number)
    fig.add_subplot(rows,columns,2)
    plt.imshow(Y)
    plt.title("Ground Truth")
    fig.add_subplot(rows,columns,3)
    spy.imshow(X,stretch=0.02,title="GT Combined",fignum=fig.number)
    plt.imshow(Y,alpha=0.4)
    fig.add_subplot(rows,columns,4)
    spy.imshow(X,stretch=0.02,title="Hybrid BN A",fignum=fig.number)
    plt.imshow(read_tif_img(prediction_dir+"/Hybrid_BN_A/output/"+ID+".tif"),alpha=0.5)
    fig.add_subplot(rows,columns,5)
    spy.imshow(X,stretch=0.02,title="Random Forest",fignum=fig.number)
    plt.imshow(read_tif_img(prediction_dir+"/RF/output/"+ID+".tif"),alpha=0.5)
    fig.add_subplot(rows,columns,6)
    spy.imshow(X,stretch=0.02,title="SVM 5",fignum=fig.number)
    plt.imshow(read_tif_img(prediction_dir+"/RBF_SVM/output/"+ID+".tif"),alpha=0.5)
    fig.suptitle("Image "+ID)
    fig.savefig(save_dir+"/"+ID+".png")
    plt.close() # did not test this line






