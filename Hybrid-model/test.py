from utils import *
import yaml
import os
import logging
from models.HybridSN import *
import csv

from PIL import Image
# ------------------------------------------Logging Function-----------------------------------------
if not(os.path.exists("logs")):
    os.mkdir("logs")
if os.path.isfile("logs/test.log"):
    os.remove("logs/test.log")
logging.basicConfig(filename="logs/test.log", format='%(asctime)s %(levelname)-8s %(message)s', 
                    level = logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# define input and target directories
with open('./configs/test.yml', 'r') as stream:
    configs = yaml.safe_load(stream)


# ------------------------------------------Read Configuration----------------------------------------
# 模型文件夹
parameter_dir = configs["parameter_dir"]
# 输出文件夹，保存(patch, label)对
input_dir = configs["input_dir"]
# Y文件夹
label_dir = configs["label_dir"]
# 输出文件夹，保存(patch, label)对
prediction_dir = configs["prediction_dir"]
# 选择需要测试的模型，并生成预测结果
test_models = configs["test_models"]
# PCA, must match train config
pca_components = configs["pca_components"]
patch_size = configs["patch_size"]
batch_size = configs["batch_size"]
SVM_patch_size = configs["SVM_patch_size"]
test_num = configs["test_num"]

if not(os.path.exists(prediction_dir)):
    os.mkdir(prediction_dir)
# -------------------------------------------Read Data and Label-----------------------------------------
# !!! 考虑去噪和SG平滑
# Prepare input data
data_files = generate_file_list(input_dir, 'hdr')
label_files = generate_file_list(label_dir, "png")
print(len(data_files))
print(len(label_files))
assert len(data_files) == len(label_files)
N = len(data_files)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("using gpu number ", torch.cuda.device_count())

models = {}
# load models
for model_name in configs["test_models"]:
    if(configs['test_models'][model_name]['test']):
        model = load_model(parameter_dir, configs['test_models'][model_name]['param'], model_name, logger, device)
        models[model_name] = model

# build csv
csv_fields = ['File','Model','TN', 'FP', 'FN', 'TP', 'Accuracy', 'Specificity', 'Sensitivity']
csv_rows = []

# Start testing
for n in range(0, N):
    if n >= test_num:
        break
    logger.info("Now processing image %s", data_files[n])
    logger.info("Now processing image %s", label_files[n])
    print("Now processing image %s", data_files[n])
    print("Now processing image %s", label_files[n])
    # read each data and true label, pad input data and make prediction based on selected model
    data = read_process_hdr_image(data_files[n] ,pca_components)
    label = read_process_tif_img(label_files[n])
    label_img = read_tif_img(label_files[n])
    padded_data = padWithZeros(data, (patch_size-1)//2)
    logger.info("padded data has shape %s", str(padded_data.shape))
    SVM_padded_data = padWithZeros(data, (SVM_patch_size-1)//2)
    logger.info("svm padded data has shape %s", str(SVM_padded_data.shape))
    # For each model
    for model_name in configs["test_models"]:
        # perform prediction only if test is turned on
        if(configs['test_models'][model_name]['test']):
            logger.info("Testing %s", model_name)
            # Load model
            model = models[model_name]
            # Make prediction
            if (model_name == 'RBF_SVM'):
                prediction_img_red = padded_img_predict(SVM_padded_data, label, SVM_patch_size, model, 
                                                   configs['test_models'][model_name]['net'],
                                                    device, logger, batch_size,prob=0)
            else:
                prediction_img_red = padded_img_predict(padded_data, label, patch_size, model, configs['test_models'][model_name]['net'],
                                                   device, logger, batch_size)
            # Obtain test statistics
            # Change 0/1
            input_file_name = os.path.basename(label_files[n])
            input_file_without_ext = os.path.splitext(input_file_name)[0]
            prediction_model_name = prediction_dir + "/" + model_name
            if not(os.path.exists(prediction_model_name)):
                os.mkdir(prediction_model_name)
                os.mkdir(prediction_model_name+"/output")
            np.save(prediction_model_name + "/output/" + input_file_without_ext, prediction_img_red)

            # graph_ill_cell_label =  np.zeros((prediction_img_red.shape[0],prediction_img_red.shape[1],3), dtype=np.uint8)
            # graph_ill_cell_label[:,:,0] = np.where(prediction_img_red==1,255,0)
            # graph_ill_cell_label[:,:,1] = np.where(prediction_img_red==2,255,0)
            # graph_ill_cell_label[:,:,1] = np.where(prediction_img_red==3,255,0)
            # prediction_img_red[prediction_img_red==1] = 255
            # prediction_img = np.zeros((prediction_img_red.shape[0],prediction_img_red.shape[1],3), dtype=np.uint8)
            # prediction_img[:,:,0] = prediction_img_red
            # prediction_img[:,:,1] = prediction_img_red
            # Save to directory
            # prediction_model_name = prediction_dir + "/" + model_name
            # if not(os.path.exists(prediction_model_name)):
            #     os.mkdir(prediction_model_name)
            #     os.mkdir(prediction_model_name+"/output")
            # output_img = Image.fromarray(graph_ill_cell_label)
            # input_file_name = os.path.basename(label_files[n])
            # output_img.save(prediction_model_name + "/output/" + input_file_name)
