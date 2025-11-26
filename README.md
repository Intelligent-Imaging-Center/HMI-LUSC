# HMI-LUSC

HMI\_LUSC is the first publicly available hyperspectral imaging(HMI) dataset for lung squamous cell carcinoma (LUSC). This repo contain label generation and training code for paper \[].

Dataset can be found in https://doi.org/10.6084/m9.figshare.30188080.v1.

# 1. Prerequiste
Python, PyTorch and related libraries. All libraries should be easily installed by pip3.

# 2. Preprocessing
The dataset downloaded contain 10 patients, where each patients have several ROI regions indicated by LUSC_ROI_i. The ROI subfolder contain raw hyperspectral images stored in hdr and dat pair format for ENVI software reading, two labels files in png corresponding to coarse and cell-level annotation, and an RGB file for the pesudo-RGB image.

You can run "preprocess/preprocess.py" file to convert the raw dataset folder into a preprocessed dataset which is more suitable for deep learning, such that hyperspectral images, labels and rgb images are in separate folders, and hyperspectral images would receive spectral smoothing, stretching and normalization into [0,1] range. The configurations needed to adjust are self-explanary with comments, and for this project's purpose you only need to adjust the DATA_SOURCE_FOLDER to the dataset path and OUTPUT_FOLDER to your designated place.

The output folder would contain the following subfolders:
- cell_labels, cell-level labels, where 0,1,2,3 are corresponding to non-cell(black), nonill-cell(red), ill cell(green) and background(blue) region. You can consider it as instance segmantation for cell.
- datacubes, preprocessed hyperspectral datacubes in [0,1] range.
- labels, coarse labels originally provided by pathologists, where 0 indicating non-ill and 1 indicating ill.
- rgbs, mainly used for visual check and neccessary for cell-level label generation interface we designed

# 3. Cell-level Label generation
## 3.1 KMean images
Use script in preprocess/Kmean-classification.py to automatically generate classification of each hyperspectral file in the input folder.

* KMEAN\_NUM, total classification numbers which is 20 by default. The values are chosen based on the tradeoff between classifications details and manual selection difficulties.
* input\_dir, folder contain hdr and dat files for hyperspectral images.
* output\_dir, output folder contain classification labels, for easier use of manual label selection, same as input\_dir by default.
  
## 3.2 QtLabels

Run preprocess/QtLabel/widget.py to use the manual label selection user interface. Instructions are already in the interface. There are several suggestions for easier usage.

* The input folder must contain hdr and dat files for hyperspectral images, and the classifications labels created earlier.
* We select cells and background separately. When you output the file, please put them into a separate output folder and name accordingly.
* Use the change mask button to switch between the classification labels and the selected labels. This allows you to check whether the selected regions satisfy your need.

Images below show examples of selecting some background regions for P2-1. 
![Select background regions - 1](./images//QtLabel_explained.png)
![Select background regions - 2](./images/QtLabel2_explained.png)

In the future we plan to further polish this user interface such that we can choose based on mouse selected points rather than visually find the corresponding regions. This should significantly accelerate the selection speed and ahcieve quick semi-automatic labels selection.

## 3.3 Label combination
Once you obtain the cell and background labels, together with the tumor label provided by physicians, you can create the four types label used in training. 

You should have 3 input folders ready
- coarse labels folder, indicating whether regions are ill or non-ill, which is provided by the original dataset and can be found in the output folder of 2. Preprocessing.
- cell folder, indicating which regions belong to cells, provided by 3.2 QtLabels generation
- background folder, indicating what areas are background, selected due to background noise is very large, provided by 3.2 Qtlabels generation or by Photoshop region selection. We do it by photoshop since background are connected with clear edges with tissue regions.

When three folders can prepared, you can run ./preprocess/label_generation.py to obtain the four class labels. We have provided ./preprocess/label_generation_sample.py for testing. These data come from the actual labels we have used in real experiments and full data can be obtained by request.    

# 4. Tumor Detection Network
To this point, you should have accessed to all required data and labels for tumor detection deep learning.
## 4.1 Usage instruction
In this experiment we use patch level diagonosis 

To train your own models, adjust files in /Hybrid-model/configs and run /Hybrid-model/preprocess.py, train.py and test.py sequentially. Variables names are self-explanatory. The train\_model parameters in train.yml and test\_model-test parameters in test.yml indicating what models are involved, where 1 means including the model and 0 means excluding the model.

Once the test.py finishes, the probability npy files are generated and we will use /Hybrid-model/postprocess.py and postprocess\_direct.py to obtain the full prediction labels. The main difference between two files is that postprocess\_direct.py will predict each pixel by its highest probability type, where postprocess.py would additionally consider adjacent prediction types and use a voting strategy to achieve instance segmentation.

# Contact

If there are other things unclear or code could be not run, please raise Issues or contact hhuang2@stu.xidian.edu.cn. We may simplify and update codes in future.

