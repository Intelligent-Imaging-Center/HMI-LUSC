
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
# Read configurations
with open('./analysis/configs.yml', 'r') as stream:
    configs = yaml.safe_load(stream)

performance_dir = configs['performance']
imgs_dir = configs['imgs']
if not(os.path.exists(imgs_dir)):
    os.mkdir(imgs_dir)

# 读取模型运行历史与结果
current_Acc_his = np.load(performance_dir + "/acc.npy")
current_Acc_his_BN = np.load(performance_dir + "/acc_BN.npy")
current_Acc_his_BN_A = np.load(performance_dir + "/acc_BN_A.npy")
current_specificity_his = np.load(performance_dir + "/spec.npy")
current_specificity_his_BN = np.load(performance_dir + "/spec_BN.npy")
current_specificity_his_BN_A = np.load(performance_dir + "/spec_BN_A.npy")
current_sensitivity_his = np.load(performance_dir + "/sens.npy")
current_sensitivity_his_BN = np.load(performance_dir + "/sens_BN.npy")
current_sensitivity_his_BN_A = np.load(performance_dir + "/sens_BN_A.npy")
loss_his = np.load(performance_dir+"/loss.npy")
loss_his_BN = np.load(performance_dir+"/loss_BN.npy")
loss_his_BN_A = np.load(performance_dir+"/loss_BN_A.npy")




RF_result = np.load(performance_dir + "/RF_result.npy")
accuracy_RF, specificity_RF, sensitivity_RF = RF_result
Linear_SVM_result = np.load(performance_dir + "/Linear_SVM_result.npy")
accuracy_SVM_linear, specificity_SVM_linear, sensitivity_SVM_linear = Linear_SVM_result
RBF_SVM_result = np.load(performance_dir + "/RBF_SVM_result.npy")
accuracy_SVM_rbf,specificity_SVM_rbf,sensitivity_SVM_rbf = RBF_SVM_result

# -------------------------------------网络性能对比--------------------------------------
# Accuracy
plt.cla()
plt.plot(current_Acc_his, color = 'red' , label = "Hybrid")
plt.plot(current_Acc_his_BN, color = 'blue' , label = "Hybrid+BN")
plt.plot(current_Acc_his_BN_A, color = 'black' , label = "Hybrid+BN+Attention")
plt.title("Accuracy Plot")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(imgs_dir+"/acc_his.png",dpi=300)
max_acc_Hy = max(current_Acc_his)
max_acc_Hy_BN = max(current_Acc_his_BN)
max_acc_Hy_BN_A = max (current_Acc_his_BN_A)

# specificity
plt.cla()
plt.plot(current_specificity_his, color = 'red' , label = "Hybrid")
plt.plot(current_specificity_his_BN, color = 'blue' , label = "Hybrid+BN")
plt.plot(current_specificity_his_BN_A, color = 'black' , label = "Hybrid+BN+Attention")
plt.title("Specificity Plot")
plt.xlabel("Epoch")
plt.ylabel("Specificity")
plt.legend()
plt.savefig(imgs_dir+"/spec_his.png",dpi=300)

max_specificity_Hy = max(current_specificity_his)
max_specificity_Hy_BN = max(current_specificity_his_BN)
max_specificity_Hy_BN_A = max (current_specificity_his_BN_A)


# sensitivity
plt.cla()
plt.plot(current_sensitivity_his, color = 'red' , label = "Hybrid")
plt.plot(current_sensitivity_his_BN, color = 'blue' , label = "Hybrid+BN")
plt.plot(current_sensitivity_his_BN_A, color = 'black' , label = "Hybrid+BN+Attention")
plt.title("Sensitivity Plot")
plt.xlabel("Epoch")
plt.ylabel("Sensitivity")
plt.legend()
plt.savefig(imgs_dir+"/sens_his.png",dpi=300)

max_sensitivity_Hy = max(current_sensitivity_his)
max_sensitivity_Hy_BN = max(current_sensitivity_his_BN)
max_sensitivity_Hy_BN_A = max (current_sensitivity_his_BN_A)

# loss
plt.cla()
plt.plot(loss_his, color = 'red' , label = "Hybrid")
plt.plot(loss_his_BN, color = 'blue' , label = "Hybrid+BN")
plt.plot(loss_his_BN_A, color = 'black' , label = "Hybrid+BN+Attention")
plt.title("Loss Plot")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(imgs_dir+"/sens_his.png",dpi=300)



# ------------------------------------------SVM与随机森林---------------------------------------------
x=[1,2,3,4,5,6]  # 确定柱状图数量,可以认为是x方向刻度
y=[max_acc_Hy,max_acc_Hy_BN,max_acc_Hy_BN_A,accuracy_RF,accuracy_SVM_linear, accuracy_SVM_rbf]  # y方向刻度
color=['red','black','peru','orchid','deepskyblue',"blue"]
x_label=['Hybrid','Hybrid+BN','Hybrid+BN+Attention','RF','Linear SVM','RBF SVM']

plt.cla()
plt.xticks(x, x_label)  # 绘制x刻度标签
plt.bar(x, y,color=color)  # 绘制y刻度标签

#设置网格刻度
plt.grid(True,linestyle=':',color='r',alpha=0.6)
plt.title("Accuracy Comparison")
# 获取当前图形对象
fig = plt.gcf()
# 设置图形的大小，单位为英寸
fig.set_size_inches(10, 4)  # 宽度为8英寸，高度为4英寸
plt.savefig(imgs_dir+"/acc.png",dpi=300)


# Specificity

x=[1,2,3,4,5,6]  # 确定柱状图数量,可以认为是x方向刻度
y=[max_specificity_Hy,max_specificity_Hy_BN,max_specificity_Hy_BN_A,specificity_RF,specificity_SVM_linear, specificity_SVM_rbf]  # y方向刻度

color=['red','black','peru','orchid','deepskyblue',"blue"]
x_label=['Hybrid','Hybrid+BN','Hybrid+BN+Attention','RF','Linear SVM','RBF SVM']
plt.cla()
plt.xticks(x, x_label)  # 绘制x刻度标签
plt.bar(x, y,color=color)  # 绘制y刻度标签

#设置网格刻度
plt.grid(True,linestyle=':',color='r',alpha=0.6)
plt.title("Specificity Comparison")
# 获取当前图形对象
fig = plt.gcf()
# 设置图形的大小，单位为英寸
fig.set_size_inches(10, 4)  # 宽度为8英寸，高度为4英寸
plt.savefig(imgs_dir+"/spec.png",dpi=300)


# Sensitivity

x=[1,2,3,4,5,6]  # 确定柱状图数量,可以认为是x方向刻度
y=[max_sensitivity_Hy,max_sensitivity_Hy_BN,max_sensitivity_Hy_BN_A,sensitivity_RF,sensitivity_SVM_linear,sensitivity_SVM_rbf]  # y方向刻度

color=['red','black','peru','orchid','deepskyblue',"blue"]
x_label=['Hybrid','Hybrid+BN','Hybrid+BN+Attention','RF','Linear SVM','RBF SVM']
plt.cla()
plt.xticks(x, x_label)  # 绘制x刻度标签
plt.bar(x, y,color=color)  # 绘制y刻度标签

#设置网格刻度
plt.grid(True,linestyle=':',color='r',alpha=0.6)
plt.title("Sensitivity Comparison")
# 获取当前图形对象
fig = plt.gcf()
# 设置图形的大小，单位为英寸
fig.set_size_inches(10, 4)  # 宽度为8英寸，高度为4英寸
plt.savefig(imgs_dir+"/sens.png",dpi=300)