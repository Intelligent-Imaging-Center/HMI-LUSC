from utils import *
import yaml
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import csv
def get_file_name(path):
    return Path(path).stem

# ------------------------------------------Logging Function-----------------------------------------
if not(os.path.exists("logs")):
    os.mkdir("logs")
if os.path.isfile("logs/preprocess.log"):
    os.remove("logs/preprocess.log")
logging.basicConfig(filename="logs/preprocess.log", format='%(asctime)s %(levelname)-8s %(message)s', 
                    level = logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# define input and target directories
with open('./configs/preprocess.yml', 'r') as stream:
    configs = yaml.safe_load(stream)

# build csv
csv_fields = ['File','Type','Max', 'Min', 'Avg', 'Std']
csv_rows = []

# ------------------------------------------Read Configuration----------------------------------------
# X文件夹
data_dir = configs["data"]
# Y文件夹
label_dir = configs["label"]
# 输出文件夹，保存(patch, label)对
output_dir = "stat"
# 使用 PCA 降维，得到主成分的数量，需要大于15
pca_components = configs["pca_components"]

# -------------------------------------------Read Data and Label-----------------------------------------
# !!! 考虑去噪和SG平滑

data_files = generate_file_list(data_dir, 'hdr')
label_files = generate_file_list(label_dir, "tif")
print("Total files available " + str(len(data_files)))
print(len(label_files))
assert len(data_files) == len(label_files)
for i in range(0,len(data_files)):
    if i == 1:
        break
    print("Processing file "+ data_files[i])
    data = read_hdr_file(data_files[i])
    label = read_process_tif_img(label_files[i])

    h = label.shape[0]
    w = label.shape[1]
    background_index = np.transpose((label==0).nonzero())
    nonill_index = np.transpose((label==1).nonzero())
    ill_index = np.transpose((label==2).nonzero())
    background = np.zeros((pca_components,4),dtype = np.float32)
    ill = np.zeros((pca_components,4),dtype = np.float32)
    nonill = np.zeros((pca_components,4),dtype = np.float32)
    for j in range(0, pca_components):
        band_data = data[:,:,j]
        band_background = band_data[background_index[:,0], background_index[:,1]]
        band_nonill = band_data[nonill_index[:,0], nonill_index[:,1]]
        band_ill = band_data[ill_index[:,0], ill_index[:,1]]
        background[j,:] = np.array([band_background.max(), band_background.min(), np.average(band_background), np.std(band_background)])
        if band_nonill.shape[0] > 0:
            nonill[j,:] = np.array([band_nonill.max(), band_nonill.min(), np.average(band_nonill), np.std(band_nonill)])
        if band_ill.shape[0] > 0 : 
            ill[j,:] = np.array([band_ill.max(), band_ill.min(), np.average(band_ill), np.std(band_ill)])
    x = np.arange(pca_components)
    # print(background)
    plt.plot(x, background[:,2], label = "background avg")
    plt.plot(x, background[:,1], label = "background min")
    plt.plot(x, background[:,0], label = "background max")
    plt.plot(x, ill[:,2], label = "ill cell avg")
    plt.plot(x, ill[:,1], label = "ill cell min")
    plt.plot(x, ill[:,0], label = "ill cell max")
    plt.plot(x, nonill[:,2], label = "non ill cell avg")
    plt.plot(x, nonill[:,1], label = "non ill cell min")
    plt.plot(x, nonill[:,0], label = "non ill cell max")
    plt.legend()

    plt.savefig(output_dir + "/" + get_file_name(data_files[i]) + ".png")
    plt.cla()
    # plt.show()
    # csv_rows.append([data_files[i], "background", background[:,0], background[:,0] ])

# #---------------------------------------------Generate Stat graph-----------------------------------------------

