from models.HybridSN import *
import logging
import yaml
import os
import numpy as np
from dataset import *
from utils import *
import torch




# ------------------------------------------Logging Function-----------------------------------------
if not(os.path.exists("logs")):
    os.mkdir("logs")
if os.path.isfile("logs/train.log"):
    os.remove("logs/train.log")
logging.basicConfig(filename="logs/train.log", format='%(asctime)s %(levelname)-8s %(message)s', 
                    level = logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Read configurations
with open('./configs/train.yml', 'r') as stream:
    configs = yaml.safe_load(stream)
    
# ------------------------------------------Read data-------------------------------------------------

lr = configs['params']['Hybrid']['lr']
gamma = configs['params']['Hybrid']['gamma']
num_epoch = configs['params']['Hybrid']['num_epoch']
lr_steps = configs['params']['Hybrid']['lr_steps']
momentum = configs['params']['Hybrid']['momentum']
parameter_dir = configs['parameter_dir']


if not(os.path.exists(parameter_dir)):
    os.mkdir(parameter_dir)
print(parameter_dir)

X_train = np.load(configs['input']+"/X_train.npy")
X_test = np.load(configs['input']+"/X_test.npy")
y_train = np.load(configs['input']+"/y_train.npy")
y_test = np.load(configs['input']+"/y_test.npy")
logger.info("Done reading data")
patch_size = X_train.shape[1]
pca_components = X_train.shape[3]
# -------------------------------------------Processing Data -----------------------------------------
# reshape into batch_size x width x height x pca_components (bands) x 1
Xtrain = X_train.reshape(-1, patch_size, patch_size, pca_components, 1)
Xtest  = X_test.reshape(-1, patch_size, patch_size, pca_components, 1)
logger.info('before transpose: Xtrain shape: %s', Xtrain.shape) 
logger.info('before transpose: Xtest  shape: %s', Xtest.shape) 

# Tranpose into batch_size x 1 x pca_components (bands) x width x height
Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
Xtest  = Xtest.transpose(0, 4, 3, 1, 2)
logger.info('after transpose: Xtrain shape: %s', Xtrain.shape) 
logger.info('after transpose: Xtest  shape: %s', Xtest.shape) 

# Create Dataset and DataLoader
trainset = TrainDS(Xtrain, y_train)
testset  = TestDS(Xtest,y_test)
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=256, shuffle=True,num_workers=0)
test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=256, shuffle=False, num_workers=0)
logger.info("Dataset already in data loader")
# -------------------------------------------Start training-----------------------------------------


# -------------------------------------------Hybrid Net-----------------------------------------
#  Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hybrid_BN_A
if (configs['train_models']['Hybrid_BN_A']):
    logger.info("Start training hybrid net with BN and attention")
    print("Start training hybrid net with BN and attention")
    net = HybridSN_BN_Attention().float().to(device)
    net,current_loss_his_BN_A,current_Acc_his_BN_A, current_specificity_his_BN_A, current_sensitivity_his_BN_A= train(net,logger,
                                                                                                                    device, train_loader,test_loader,lr,num_epoch,lr_steps,gamma)
    torch.save(net.state_dict(), parameter_dir + '/hybrid_BN_A.pth')
    del net
    torch.cuda.empty_cache()

if (configs['train_models']['CNN2D']):
    logger.info("Start training 2D CNN")
    print("Start training 2D CNN")
    net = CNN2D().float().to(device)
    net,current_loss_his_BN_A,current_Acc_his_BN_A, current_specificity_his_BN_A, current_sensitivity_his_BN_A= train(net,logger,
                                                                                                                    device, train_loader,test_loader,lr,num_epoch,lr_steps,gamma)
    torch.save(net.state_dict(), parameter_dir + '/cnn_2d.pth')
    del net
    torch.cuda.empty_cache()

if (configs['train_models']['CNN3D']):
    logger.info("Start training 3D CNN")
    print("Start training 3D CNN")
    net = CNN3D().float().to(device)
    net,current_loss_his_BN_A,current_Acc_his_BN_A, current_specificity_his_BN_A, current_sensitivity_his_BN_A= train(net,logger,
                                                                                                                    device, train_loader,test_loader,lr,num_epoch,lr_steps,gamma)
    torch.save(net.state_dict(), parameter_dir + '/cnn_3d.pth')
    del net
    torch.cuda.empty_cache()


# ----------------------------------RF and SVM-----------------------------
if (configs['train_models']['RF'] or configs['train_models']['RBF_SVM']):
    RF_X_train = X_train.reshape(X_train.shape[0],-1)
    RF_X_test = X_test.reshape(X_test.shape[0],-1)
    import joblib

# Random Forest
if(configs['train_models']['RF']):
    n_estimators = configs['params']['RF']['n_estimators']
    max_depth = configs['params']['RF']['max_depth']
    # Check if cuml and cupy available
    logger.info("Start training Random Forest Classifier")
    print("Start training RF")
    model = None
    if(configs['cupy']):
        # from cuml.dask.ensemble import RandomForestClassifier
        from cuml.ensemble import RandomForestClassifier
        import cupy as cp
        model = RandomForestClassifier()  # 可根据需要调整参数
        model.fit(RF_X_train.astype(np.float32), y_train)
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, random_state=42)  # Adjust parameters as needed
        model.fit(RF_X_train, y_train)
    logger.info("Done training")
    print("Done training")
    joblib.dump(model, parameter_dir+"/RF.joblib", compress=3)
    RF_Y_pred = model.predict(RF_X_test.astype(np.float32))
    accuracy_RF, specificity_RF, sensitivity_RF = predict_report(y_test, RF_Y_pred, logger)
    RF_result = np.array([accuracy_RF, specificity_RF, sensitivity_RF])
    logger.info("Accuracy is %s", accuracy_RF)
    logger.info("Specificity is %s", specificity_RF)
    logger.info("Sensitivity is %s", sensitivity_RF)
    
# RBF SVM
if(configs['train_models']['RBF_SVM']):
    if(configs['cupy']):
        from cuml.svm import SVC
        import cupy as cp
        cp.cuda.Device(1).use()
    else:
        from sklearn.svm import SVC
    svm_rbf = SVC(kernel = 'rbf')    # Allow to switch to other kernels such as 'linear', 'poly', 'sigmoid'

    # Train SVM
    print("Start training RBF SVM")
    logger.info("Start training RBF SVM")
    svm_rbf.fit(RF_X_train, y_train)
    print("Done Training RBF SVM")
    logger.info("Done training RBF SVM")
    joblib.dump(svm_rbf, parameter_dir + "/RBF_SVM.joblib")
    print("dump finished")
    # Predict SVM
    SVM_rbf_y_pred = svm_rbf.predict(RF_X_test)
    print("start predicting")
    accuracy_SVM_rbf, specificity_SVM_rbf, sensitivity_SVM_rbf = predict_report(y_test, SVM_rbf_y_pred, logger)
    rbf_SVM_result = np.array([accuracy_SVM_rbf, specificity_SVM_rbf, sensitivity_SVM_rbf])
    
    logger.info("Accuracy is %s", accuracy_SVM_rbf)
    logger.info("Specificity is %s", specificity_SVM_rbf)
    logger.info("Sensitivity is %s", sensitivity_SVM_rbf)

