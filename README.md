# HMI-LUSC

HMI_LUSC is the first publicly available hyperspectral imaging(HMI) dataset for lung squamous cell carcinoma (LUSC). This repo contain label generation and training code for paper []. 

Dataset can be found in https://doi.org/10.6084/m9.figshare.30188080.v1. 

# Prerequiste
Python, PyTorch and related libraries. All libraries should be easily installed by pip3.

# Cell-level Label generation
## KMean images
Use script in /KMeanClassification/Kmean-classification.py to automatically generate classification of each hyperspectral file in the input folder.
- KMEAN_NUM, total classification numbers which is 20 by default.
- input_dir, folder contain hdr and dat files for hyperspectral images.
- output_dir, output folder contain classification labels, for easier use of manual label selection, same as input_dir by default.

## QtLabels
Run /QtLabel/widget.py to use the manual label selection user interface. Instructions are already in the interface. There are several suggestions for easier usage.
- The input folder must contain hdr and dat files for hyperspectral images, and the classifications labels created earlier.
- We select cells and background separately. When you output the file, please put them into a separate output folder and name accordingly.
- Use the change mask button to switch between the classification labels and the selected labels.

## Label combination
Once you obtain the cell and background labels, together with the tumor label provided by physicians, you can create the four types label used in training. The label generation files are not provided since the final cell-level labels are provided in the dataset.

# Tumor Detection Network
## Usage instruction
To train your own models, adjust files in /HybridV4/configs and run /HybridV4/preprocess.py, train.py and test.py sequentially. Variables names are self-explanatory. The train_model parameters in train.yml and test_model-test parameters in test.yml indicating what models are involved, where 1 means including the model and 0 means excluding the model. 

Once the test.py finishes, the probability npy files are generated and we will use /HybridV4/postprocess.py and postprocess_direct.py to obtain the full prediction labels. The main difference between two files is that postprocess_direct.py will predict each pixel by its highest probability type, where postprocess.py would additionally consider adjacent prediction types and use a voting strategy to achieve instance segmentation.

