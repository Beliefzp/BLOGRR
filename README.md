# Unsupervised Brain Tumor Segmentation via Bi-Level Optimization Guided by Radiological Reports
Official PyTorch implementation for our MICCAI 2025 workshop paper: "Unsupervised Brain Tumor Segmentation via Bi-Level Optimization Guided by Radiological Reports"

## Abstract
Unsupervised brain tumor segmentation can aid brain tumor diagnosis and treatment without the high cost of manual annotations. Existing methods typically use a reconstruction-based strategy, where an image self-reconstruction network is trained with normal data and applied to images with brain tumors. The reconstruction error map is then used to indicate the tumor regions and is thresholded to obtain tumor segmentation. However, optimal threshold selection is challenging without annotations in the unsupervised case, which limits the accuracy and applicability of these reconstruction-based methods. To address the problem, in this work we propose the Bi-Level Optimization Guided by Radiological Reports (BLOGRR) framework for unsupervised brain tumor segmentation. BLOGRR extends the reconstruction-based strategy with an additional threshold estimation network. Instead of selecting an empirical fixed threshold, it determines an adaptive threshold for every sample. Specifically, we develop an iterative bi-level optimization procedure, where lower and upper loops jointly update the reconstruction network and threshold estimation network. As no manual annotation is available, BLOGRR resorts to radiological reports, which provide key descriptions of image anomalies in the form of natural language, for learning the threshold determination. The reports are processed with brain anatomical knowledge to indicate potential tumor regions. Two loss functions are developed for the two loops to optimize the reconstruction network and threshold estimation network. Experimental results on a public dataset and an in-house dataset indicate that BLOGRR outperforms existing unsupervised methods with noticeable improvements.

## Experimental results
<p align="center"> <img src="imgs/main_result.png" width="80%"> </p>

## Training
### Data Preparation
#### First Stage
We begin by registering all the datasets used in the study—including HCP, BraTS, and the in-house dataset—to the MNI152 standard space.

#### Second Stage
Since BLOGRR currently only supports 2D images, while most medical data comes in 3D volumes, the second step involves converting each 3D volume into 2D slices along the axial plane. Given that the initial and final slices of medical scans often contain little to no meaningful information, we recommend selecting slices between indices 30 and 120. As a result, each 3D volume will yield 120 2D slice images. These images are then resized to a resolution of 128×128 pixels.

#### Third Stage
Finally, the processed data is organized into specific directories. The folder structure for the datasets used in our experiments is as follows: **final_test_data** contains the BraTS2021 dataset used for final testing; **HCP_train_data** and **In_house_data** contain the training data for the BLOGRR model; **sample_test_data** is a small dataset used to monitor the BLOGRR training process in real time, which can be created by randomly sampling a subset of BraTS2021 data.
```
├── Dataset
│   ├── final_test_data
│   │   ├── img
│   │   └── seg
│   ├── HCP_train_data
│   │   └── img
│   ├── In_house_data
│   │   ├── img
│   │   └── seg
│   ├── sample_test_data
│   │   ├── img
│   │   └── seg
```

### Start Training
#### First Stage
在准备好数据集以后，你需要修改以后**dataload.py**中的数据集加载代码，这个比较简单，根据你的数据命名方式简单修改一下即可，相信你可以的！
同时，如果你将数据集放在了一些自定义的文件夹下面，没有按照**Data Preparation**部分所介绍的文件夹格式的话，你需要修改一下**BLOGRR.py**中**get_config**函数中的数据集路径。

#### Second Stage
运行如下命令即可开始进行训练，在训练过程中，你可以通过wandb来观察训练的效果，可以手动选择较好的模型进行保存，也可以直接选择最后一个epoch的结果，不过建议选择通过观察wandb来选择较好的模型进行保存，以便可以复现出论文中的实验结果。
```
python BLOGRR.py
```



