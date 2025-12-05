# HMI-LUSC

HMI\_LUSC is the first publicly available hyperspectral imaging(HMI) dataset for lung squamous cell carcinoma (LUSC), which contain 62 hyperspectral images from 10 patients, with spatial resolution of 3088 x 2064 pixels and spectral resolution of 450-750 across 61 bands. The main purpose of this dataset is to facilitate hyperspectral diagnosis research in microscopy and specifically lung cancers. 

Dataset can be found in https://doi.org/10.6084/m9.figshare.30188080.v1, together with more detailed dataset structure and description.

![Dataset Visualization](./images/fig6.jpg)

# Environment Requirement
Python, PyTorch matplot, cv2, sklearn and related libraries. All libraries should be easily installed by pip3. You can use requirement.txt to install all required packages
```python
pip install -r requirements.txt
```

# Project Structure
This repository contains 3 components, DataPreparation, SampleTraining and CustomLabelGeneration.

In the DataPreparation folder, a single python file named preprocess.py can read files from the original dataset, redistribute them into folder formats easier for machine learning pipeline, and performs denoising, smoothing and normalizations for datacubes.

In the SampleTraining folder, we have provided a simple deep learning usage examples for this dataset, where segmentation based on patch classification is used to diagnose tumors. Although models are simple and the performance only achieve basis satifaction, they can be used as comparison benchmark for future.

In the CustomLabelGeneration folder, users can follow the pipeline we have used to convert 2 types (tumor/non-tumor) labels provided by pathologists to 4 types (non-cell, non-tumor cells, tumor cells and background) labels. To do it, a Kmean classifications is done, then users can use our custom label preparation GUI to select cell regions, and use label_generator.py to create new labels. 

# 2. DataPreparation
You can run "preprocess/preprocess.py" file to convert the raw dataset folder into a preprocessed dataset which is more suitable for deep learning, such that hyperspectral images, labels and rgb images are in separate folders, and hyperspectral images would receive spectral smoothing, stretching and normalization into [0,1] range.

There are folders parameters you must adjust. 
- DATA_SOURCE_FOLDER, which should be the Root folder containing P1, P2 you downloaded from the website
- OUTPUT_FOLDER, default to "processed_dataset", which is the output root folder ready for deep learning pipeline

There are hyperspectral cubes preprocess parameters you can adjust.
PERCENT_LINEAR_VALUE = 0  # Percentile for linear stretch (e.g., 2 for 2%)
PERBAND = 0               # 1: Process each band independently; 0: Process 3D cube as a whole.
SNV_USED = 1              # 1: Apply Standard Normal Variate; 0: Off
OPTLIN = 1                # 1: Apply Optimized Linear Stretch; 0: Off (Uses standard percent linear)

Simple go to the subfolder and run the program.
```python
cd DataPreparation
python preprocess.py
```


 The configurations needed to adjust are self-explanary with comments, and for this project's purpose you only need to adjust the DATA_SOURCE_FOLDER to the dataset path and OUTPUT_FOLDER to your designated place.

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
- background folder, indicating what areas are background, selected due to background noise is very large, provided by 3.2 Qtlabels generation or other region selection softwares. 

When three folders can prepared, you can run ./preprocess/label_generation.py to obtain the four class labels. We have provided ./preprocess/label_generation_sample.py for testing. These data come from the actual labels we have used in real experiments and full data can be obtained by request.    

# 4. Tumor Detection Network
To this point, you should have accessed to all required data and labels for tumor detection deep learning.
## 4.1 Usage instruction
In this experiment we use patch level diagonosis to perform pixel-level segmentation. In other word, we will extract a small patch from one pixel as one sample data and let the pixel label as the patch label. We will extract patches from all available data in the training set on a balanced types basis, where the ratio for 0123 is 2:1.5:0.5:1. The variable patch_stride is a deprecated variable and no longer effective.

To train your own models, adjust files in /Hybrid-model/configs and run /Hybrid-model/preprocess.py, train.py and test.py sequentially. Variables names are self-explanatory. The train\_model parameters in train.yml and test\_model-test parameters in test.yml indicating what models are involved, where 1 means including the model and 0 means excluding the model.

There are 5 models to choose from, where the first three CNN models are defined in models/HybridSN
- HybridSN: modified from ...
- CNN3D: A simple implementation of CNN3D.
- CNN2D: A simple implementation of CNN2D.
- RF (random forest): sklearn implementation with default value...
- SVM (support vector machine): sklearn implementation with default value...

Once the test.py finishes, the probability npy files are generated and we will use /Hybrid-model/postprocess.py and postprocess\_direct.py to obtain the full prediction labels. The main difference between two files is that postprocess\_direct.py will predict each pixel by its highest probability type, where postprocess.py would additionally consider adjacent prediction types and use a voting strategy to achieve instance segmentation.

# Contact
If the dataset can no longer be accessed, codes cannot be run or other things raise your concern, please raise Issues or contact hhuang2@stu.xidian.edu.cn (available before May 2028). We may simplify and update codes in future.

# Citation 
If you use HMI-LUSC in your research, please cite the paper (details updated in future since the paper is under review).

```bibtex
@article{yan2025LUSC,
title = {},
author = {Zhiliang Yan, Haosong Huang, Yunfeng Nie},
journal={},
year={2026}
}
```