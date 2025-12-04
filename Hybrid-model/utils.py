import numpy as np
import math
import os
import copy
# model related
from sklearn.model_selection import train_test_split
import scipy.signal
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
# IO related
from spectral import *
import imageio.v2 as iio
import joblib 
# custum library
from dataset import *
from models.HybridSN import *
import tqdm
# -------------------------------------Read image and model--------------------------------------------
def generate_file_list(dir, end):
    list = [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith(end)]
    list.sort()
    return list

# Perform PCA transformation for hyperspectral data X
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

# Change HDR file into an image array with 69 bands 
def read_hdr_file(file):
    return np.array(scipy.signal.savgol_filter(open_image(file).load(),5,2))
    # return np.array(open_image(file).load())
    # return np.array(open_image(file).load())[:,:,30:31]

# Process data, including PCA and normalization
def process_hdr_image(img, pca_components):
    # print("before PCA ", img.min(), img.max())
    PCA_img = applyPCA(img, pca_components)
    return PCA_img.astype(np.float32)

def read_process_hdr_image(file, pca_components):
    img = read_hdr_file(file)
    return process_hdr_image(img, pca_components)

# read and process label tif image, only keep red channel and change[0, 255] to [0, 1]
def read_tif_img(file):
    return np.array(iio.imread(file), dtype = np.uint8)

def process_tif_img(img):
    if(len(img.shape)>2):
        r_img = img[:,:,0]
        r_img[r_img==255]=1
        return r_img
    else:
        return img

def read_process_tif_img(file):
    return process_tif_img(read_tif_img(file))

# Load model from parameter file
def load_model(param_dir, param_file, model_name, logger, device, **kwargs):
    param_file = param_dir+"/"+param_file
    max_type = kwargs.get("max_type", None)  
    if model_name == "Hybrid_BN_A":
        if max_type is None:
            model = HybridSN_BN_Attention()
        else:
            model = HybridSN_BN_Attention(classification_types_num = max_type)
        state_dict = torch.load(param_file)
        remove_prefix = 'module.'
        state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        logger.info("correctly load Hybrid BN A")
        model.to(device)
        model = torch.nn.DataParallel(model)
    elif model_name == "CNN2D":
        if max_type is None:
            model = CNN2D()
        else:
            model = CNN2D(classification_types_num = max_type)
        state_dict = torch.load(param_file)
        remove_prefix = 'module.'
        state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        logger.info("correctly load CNN2D")
        model.to(device)
        model = torch.nn.DataParallel(model)
    elif model_name == "CNN3D":
        if max_type is None:
            model = CNN3D()
        else:
            model = CNN3D(classification_types_num = max_type)
        state_dict = torch.load(param_file)
        remove_prefix = 'module.'
        state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        logger.info("correctly load CNN3D")
        model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model = joblib.load(param_file)
    return model

# ----------------------------------- Processing raw image ------------------------------------------------
# Perform zero padding for pixels at image edges when extracting patches
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def generatePatch(X, r, c, margin):
    return X[r - margin:r + margin + 1, c - margin:c + margin + 1]


def generateIndexPairByStride(h,w, stride):
    XIndex = []
    YIndex = []
    for r in range(0, h, stride):
        for c in range(0, w, stride):
            XIndex.append(r)
            YIndex.append(c)
    return np.array(XIndex), np.array(YIndex)

def euclid_dist(a,b):
    return math.abs(a[0]-b[0])+math.abs(a[1]-b[1])

def generateIndexPair(Y, stride, h, w):
    # 2, 0.5, 1.5, 1
    # 5 in total, each type 200
    np.random.seed(42)
    each_type_num = 200
    Y = Y.reshape((h,w))
    print(np.unique(Y))
    non_ill_cell = np.transpose((Y==1).nonzero())
    ill_cell = np.transpose((Y==2).nonzero())
    non_cell = np.transpose((Y==0).nonzero())
    background = np.transpose((Y==3).nonzero())
    # print("Stride generated patches are ", StrideIndex.shape)
    # print("Cell generated patches are ", cell.shape)
    np.random.shuffle(non_cell)
    non_cell = non_cell[:each_type_num*2,:]
    np.random.shuffle(ill_cell)
    ill_cell = ill_cell[:each_type_num//2,:]
    np.random.shuffle(non_ill_cell)
    non_ill_cell = non_ill_cell[:each_type_num//2*3,:]
    np.random.shuffle(background)
    background = background[:each_type_num,:]
    # print("Cell has " + str(cell.shape[0]) + " non-cell has " + str(StrideIndex.shape[0]))
    return np.vstack((ill_cell,non_ill_cell,non_cell, background))

# Y is n x h x w
def generateIndexPairMultipleWrapper(Y, stride):
    h = Y.shape[1]
    w = Y.shape[2]
    Y = Y.reshape(Y.shape[0],-1)
    data = np.apply_along_axis(generateIndexPair,1,Y,stride,h,w )
    return data

# Extract patches from image X based on given XIndex and YIndex
# Input: Single hyperspectral image X (H,W,B) and preprocessed label image y (H,W), windowSize is the patch size
# Output: patches（H*W，window，window，B），label（H*W，1）
def getPatchesXFromImage(X, XIndex, YIndex, windowSize=25, padded = False):
    margin = int((windowSize - 1) / 2)
    if not padded:
        zeroPaddedX = padWithZeros(X, margin)
    N = XIndex.shape[0]
    # split patches
    patchesData = np.zeros((N, windowSize, windowSize, X.shape[2]),dtype=np.float32)
    for i in range(0, N):
        r = XIndex[i]
        c = YIndex[i]
        if padded:
            patchesData[i, :, :, :] = generatePatch(X,r+margin,c+margin,margin) 
        else: 
            patchesData[i, :, :, :] = generatePatch(zeroPaddedX,r+margin,c+margin,margin) 
    return patchesData

def getPatchesYFromImage(y, XIndex, YIndex):
    N = XIndex.shape[0]
    patchesLabels = np.zeros((N),dtype=np.uint8)
    for i in range(0, N):
        r = XIndex[i]
        c = YIndex[i]
        patchesLabels[i] = y[r,c]
    return patchesLabels

# Split dataset into training and testing set
def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=int)
    y_test = np.array(y_test, dtype=int)
    return X_train, X_test, y_train, y_test

# ----------------------------------------- Testing and Metrics related -------------------------------------------------------
 
# In order accuracy, specificity, sensitivity
def predict_report(y_true, y_pred, logger):
    matrix = confusion_matrix(y_true, y_pred,labels=[0,1,2,3])
    tp = np.diag(matrix)
    fp = np.sum(matrix,axis=0) - tp
    fn = np.sum(matrix,axis=1) - tp
    tn = matrix.sum() - tp - fp - fn
    # logger.info("Prediction has tn %s fp %s fn %s tp %s", str(tn), str(fp), str(fn), str(tp))
    return (tn+tp)/(tn+fp+fn+tp), tn/(tn+fp), tp/(tp+fn)

def test_acc(net, test_loader, device, logger, findMax = True):
    count = 0
    net.eval()
    with torch.no_grad():
        for inputs, label in test_loader:
            inputs = inputs.to(device)
            outputs = net(inputs.float())
            if findMax:
                outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            else:
                outputs = outputs.detach().cpu().numpy()
            if count == 0:
                y_pred_test =  outputs
                count = 1
                y_true = label
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_true =np.concatenate((y_true, label))

    # generate classifcation report
    if findMax:
        accuracy, specificity, sensitivity = predict_report(y_true, y_pred_test, logger)
    else:
        accuracy, specificity, sensitivity = [0,0,0]
    return float(np.mean(accuracy)), float(np.mean(specificity)), float(np.mean(sensitivity)), y_pred_test


# ----------------------------------------- Training and prediction related ---------------------------------------------------
def toNetPatch(patch):
    patch_size = patch.shape[1]
    pca_components = patch.shape[3]
    patch  = patch.reshape(-1, patch_size, patch_size, pca_components, 1)
    return patch.transpose(0, 4, 3, 1, 2)

# Should start from preprocessed data (patch, Y)
def patch_predict(patch, Y, margin, model, net, device, logger, batch_size,prob=1):
    if (net):
        patch = toNetPatch(patch)
        XDataset = TestDS(patch,Y)
        input_loader = torch.utils.data.DataLoader(dataset=XDataset, batch_size=batch_size, shuffle=False,num_workers=0)
        acc, spe, sen, y_prediction = test_acc(model, input_loader, device, logger, False)
        # logger.info("Prediction has acc %s spe %s sen %s", str(acc), str(spe), str(sen))
        # logger.info("prediction has shape %s", str(y_prediction.shape))
        return y_prediction
    else:
        if prob:
            result = model.predict_proba(patch.reshape(patch.shape[0],-1))
        else:
            result = model.predict(patch.reshape(patch.shape[0],-1))
        return result

# input
# X: h x w x b
# Y: (hxw)
def padded_img_predict(X, Y, windowSize, model, net, device, logger, batch_size,prob=1):
    
    patchStride = 1
    margin = (windowSize-1)//2
    logger.info("margin is %s", str(margin))
    output = np.array([])
    first = True
    XIndex, YIndex = generateIndexPairByStride(Y.shape[0], Y.shape[1], 1)
    IndexDataset = IndexDS(XIndex,YIndex)
    index_loader = torch.utils.data.DataLoader(dataset=IndexDataset, batch_size=batch_size, shuffle=False,num_workers=0)
    for rows, cols in tqdm.tqdm(index_loader):
        # logger.info("Now we have % th batch")
        batchPatch = getPatchesXFromImage(X,rows,cols,windowSize,True)
        batchY = getPatchesYFromImage(Y,rows,cols)
        prediction = patch_predict(batchPatch, batchY,margin, model, net, device, logger, batch_size,prob=prob)
        # print("output   and prediction shape are ", output.shape, prediction.shape)
        if first:
            output= prediction
            first = False
        else:
            output = np.concatenate((output,prediction))
    if prob:
        return output.reshape((Y.shape[0],Y.shape[1],4))
    else:
        return output.reshape((Y.shape[0],Y.shape[1]))

def train(net, logger, device, train_loader, test_loader, lr = 0.001, num_epoch=30, lr_steps=[20,40], gamma=0.1):
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        logger.info("\n\nLet's use %s GPUs!\n\n", str(torch.cuda.device_count()))
    if len(device_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=device_ids)

    current_loss_his = []
    current_Acc_his = []
    current_specificity_his = []
    current_sensitivity_his = []

    best_net_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    criterion = torch.nn.CrossEntropyLoss()
#     optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr, momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, gamma)
   
    # Start training
    total_loss = 0
    for epoch in range(num_epoch):
        net.train()  # 将模型设置为训练模式
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 正向传播 +　反向传播 + 优化 
            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # scheduler.step()
        current_acc, current_specificity, current_sensitivity, y_pred_test = test_acc(net, test_loader, device, logger)
        current_Acc_his.append(current_acc)
        current_specificity_his.append(current_specificity)
        current_sensitivity_his.append(current_sensitivity)
        if current_acc > best_acc:
            best_acc = current_acc
            best_net_wts = copy.deepcopy(net.state_dict())
        logger.info('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]  [current acc: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item(), current_acc))
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]  [current acc: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item(), current_acc))
        current_loss_his.append(loss.item())
    logger.info('Finished Training')
    print("Finish training")
    logger.info("Best Acc:%.4f" %(best_acc))
    # load best model weights
    net.load_state_dict(best_net_wts)

    return net,current_loss_his,current_Acc_his, current_specificity_his, current_sensitivity_his