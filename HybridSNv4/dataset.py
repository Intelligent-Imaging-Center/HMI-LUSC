import torch

""" Training dataset"""
class TrainDS(torch.utils.data.Dataset): 
    def __init__(self, Xtrain, y_train):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(y_train)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset): 
    def __init__(self, Xtest, y_test):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(y_test)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len
    
""" Index tuple dataset"""
class IndexDS(torch.utils.data.Dataset): 
    def __init__(self, XIndex, YIndex):
        self.len = XIndex.shape[0]
        self.x_index = torch.LongTensor(XIndex)
        self.y_data = torch.LongTensor(YIndex)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_index[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len