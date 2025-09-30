import os
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt

data = spy.open_image().load()

plt.rcParams['font.sans-serif'] = ['SimHei']
hsi_img = spy.imshow(data, stretch=0.02, title='高光谱图像_双击图像中的点查看光谱曲线\n按‘shift+左键’选择感兴趣区域')
plt.show(block=True)