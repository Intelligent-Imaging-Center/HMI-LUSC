from utils import *
from models.HybridSN import *
from PIL import Image
import spectral as spy
import matplotlib.pyplot as plt

# spy.imshow(read_hdr_file("../data/1.hdr"),stretch=0.02,title="1-reverrrrrrrrrse")
# spy.imshow(read_hdr_file("../data200/1.hdr"),stretch=0.02,title="14-r")
# plt.imshow(read_tif_img("../pred_label6_1/Hybrid_BN_A/output/11.tif"),alpha=0.5)
# spy.imshow(read_hdr_file("../data/15.hdr"),stretch=0.02,title="15")
# spy.imshow(read_hdr_file("../large_data/[裁剪] lin6-1_r3c6.hdr"),stretch=0.02,title="r1c1")
# spy.imshow(read_hdr_file("../large_data/[裁剪] lin6-1_r1c2.hdr"),stretch=0.02,title="r1c2")
X = read_hdr_file("../RawData/data-large/lin6-3.hdr")
print(X.max())
print(X.min())
# spy.imshow(X,(55, 24, 3), stretch=0.02,title="24")
spy.imshow(X,(55, 24, 3), stretch=0.02,title="24")
# plt.imshow(read_tif_img("../Prediction/large/Hybrid_BN_A/output/lin6-6.tif"),alpha=0.5)
# plt.imshow(read_tif_img("../Prediction/10-1/output.png"),alpha=0.5)
# plt.imshow(read_tif_img("../CreatingLabels/Cell/lin6-3.tif"),alpha=0.8)
plt.show(block=True)
