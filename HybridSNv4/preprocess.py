from utils import *
import yaml
import logging

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

# ------------------------------------------Read Configuration----------------------------------------
# X文件夹
data_dir = configs["data"]
# Y文件夹
label_dir = configs["label"]
# 输出文件夹，保存(patch, label)对
output_dir = configs["output"]
# 用于测试样本的比例
test_ratio = configs["test_ratio"]
# 每个像素周围提取 patch 的尺寸, 必须是奇数
patch_size = configs["patch_size"]
# 使用 PCA 降维，得到主成分的数量，需要大于15
pca_components = configs["pca_components"]
# patch步长
patch_stride = configs["patch_stride"]

# -------------------------------------------Read Data and Label-----------------------------------------
# !!! 考虑去噪和SG平滑

data_files = generate_file_list(data_dir, 'hdr')
label_files = generate_file_list(label_dir, "tif")
print("Total files available " + str(len(data_files)))
print(len(label_files))
assert len(data_files) == len(label_files)
for i in range(0,len(data_files)):
    print("Processing file "+ data_files[i])
    data = read_process_hdr_image(data_files[i], pca_components)
    print("Processing file "+ label_files[i])
    label = read_process_tif_img(label_files[i])
    h = label.shape[0]
    w = label.shape[1]
    print("Label has unique value " ,np.unique(label))
    IndexForImage = generateIndexPair(label.reshape(label.shape[0],-1), patch_stride, h,w)
    print("Done generating index")
    print("Index shaep ",IndexForImage.shape )
    singleX = getPatchesXFromImage(data,IndexForImage[:,0],IndexForImage[:,1] , patch_size)
    singleY = getPatchesYFromImage(label, IndexForImage[:,0],IndexForImage[:,1])
    if i == 0:
        X = singleX
        Y = singleY
    else:
        X = np.concatenate((X,singleX))
        Y = np.concatenate((Y,singleY))
        print(X.shape)
        print(Y.shape)

#---------------------------------------------Generate Patches-----------------------------------------------

logger.info("X shape before %s", X.shape)
# patch_data = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2],X.shape[3],X.shape[4]))
patch_data = X
logger.info("X shape after %s", patch_data.shape)
logger.info("Y shape before %s", Y.shape)
# patch_label = np.reshape(Y,(Y.shape[0]*X.shape[1]))
patch_label = Y
logger.info("Y shape after %s", patch_label.shape)

# 获得最终的训练集和测试集，保存到硬盘
X_train, X_test, y_train, y_test = splitTrainTestSet(patch_data, patch_label, test_ratio)
logger.info("Train size %s", X_train.shape[0])
logger.info("Test size %s", X_test.shape[0])

logger.info("Test contain 0 %s", np.count_nonzero(y_test==0))
logger.info("Test contain 1 %s", np.count_nonzero(y_test==1))
logger.info("Test contain 2 %s", np.count_nonzero(y_test==2))
logger.info("Test contain 3 %s", np.count_nonzero(y_test==3))
logger.info("Train contain 0 %s", np.count_nonzero(y_train==0))
logger.info("Train contain 1 %s", np.count_nonzero(y_train==1))
logger.info("Train contain 2 %s", np.count_nonzero(y_train==2))
logger.info("Train contain 3 %s", np.count_nonzero(y_train==3))




if not(os.path.exists(output_dir)):
    os.mkdir(output_dir)
logger.info("Save to %s", output_dir)
np.save(output_dir+"/X_train.npy", X_train)
np.save(output_dir+"/X_test.npy", X_test)
np.save(output_dir+"/y_train.npy", y_train)
np.save(output_dir+"/y_test.npy", y_test)
logger.info("Data saved")