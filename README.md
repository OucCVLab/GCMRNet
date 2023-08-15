# GCMFNet：Camouflaged Object Detection via Global-edge Context and Mixed-scale Refinement


> **Authors:** 
> Qilun Li, Fengqin Yao , Xiandong Wang , ShengkeWang * 

## 1. Proposed Baseline

### 1.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single NVIDIA GeForce GPU of 24 GB Memory.

1. Configuring your environment (Prerequisites):
   
    + Creating a virtual environment in terminal: `conda create -n BGNet python=3.6`.
    
    + Installing necessary packages: `pip install -r requirements.txt`.

1. Downloading necessary data:

    + downloading testing dataset and move it into `./data/TestDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view?usp=sharing).
    
    + downloading training dataset and move it into `./data/TrainDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing).
    
    + downloading pretrained weights and move it into `./checkpoints/best/BGNet.pth`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1uRsku9-ILQvNhqAYYar3O71ENhbBU1BA/view?usp=sharing).  
    
    + downloading Res2Net weights and move it into `./models/res2net101_v1b_26w_4s-0812c246.pth`[download link (Google Drive)](https://drive.google.com/file/d/1vLrr82_SbNY__etRcc4NuxODu13OIeLa/view?usp=sharing). 
   
1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `etrain.py`.

1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `etest.py` to generate the final prediction map: replace your trained model directory (`--pth_path`).

### 1.2 Evaluating your trained model:

One-key evaluation is written in MATLAB code (revised from [link](https://github.com/DengPingFan/CODToolbox)), 
please follow this the instructions in `./eval/main.m` and just run it to generate the evaluation results in.

If you want to speed up the evaluation on GPU, you just need to use the [efficient tool](https://github.com/lartpang/PySODMetrics) by `pip install pysodmetrics`.

Assigning your costumed path, like `method`, `mask_root` and `pred_root` in `eval.py`.

Just run `eval.py` to evaluate the trained model.

> pre-computed maps of GCMFNet can be found in [download link (Google Drive)](https://drive.google.com/file/d/141-7rrpmgj1f6tau6p8Plb2oEJCTTl79/view?usp=sharing). 



