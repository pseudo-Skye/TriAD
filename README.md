<p align="center" width="50%">
<img width="25%" src = "https://github.com/pseudo-Skye/TriAD/assets/117964124/d7bee644-3dbc-4547-ad78-a2fe6a0139e5">
</p>

# Unraveling the 'Anomaly' in Time Series Anomaly Detection: A Self-supervised Tri-domain Solution (ICDE 2024)

This paper addresses key challenges in time series anomaly detection (TSAD): **(1)** the scarcity of labels, and **(2)** the diversity in anomaly types and lengths. Furthermore, it contributes to the ongoing debate on the effectiveness of deep learning models in TSAD, highlighting the problems of flawed benchmarks and ill-posed evaluation metrics. This study stands out as **the first to reassess the potential of deep learning in TSAD**, employing both **rigorously designed datasets (UCR Archive)** and **evaluation metrics (PA%K and affiliation)**. ([paper](https://arxiv.org/pdf/2311.11235.pdf))

![image](https://github.com/pseudo-Skye/TriAD/assets/117964124/dcacb90f-a395-42cf-866c-a75600d40c5e)

## Overview
1. Download the [UCR dataset](https://github.com/pseudo-Skye/Data-Smith/blob/master/TSAD%20Dataset/cleaned_dataset/cleaned_dataset.zip) ready for use. Next, run **preprocess_data.py**. This script will partition 10% of the training data as the validation set and create a directory containing the dataset at **./dataset/ucr_data.pt** in the following format:
   ```
   {'train_data': train_x,
   'valid_data': valid_x,
   'test_data': test_x,
   'test_labels': test_y}
   ```

2. Simply run **train.py** to train TriAD over the whole dataset. The results are saved as **tri_res.pt** (a demo version provided) and wrapped in a data frame. 
3. To get a summary of both the tri-window and single window detection **accuracy** (among the 250 datasets, how many are successfully detected by tri/single window), simply run **single_window_selection.py**. The results will be saved as **merlin_win.pt**, which can generate the [Merlin](https://github.com/pseudo-Skye/Time-Matters/blob/main/anomaly%20detection/MERLIN%20(ICDM%2020).md) readable files by **discord_data_prep.py**. By restricting our focus to the single window, we force Merlin to scan around the window to find anomalies. 
4. To get the summary of detection results of the shortest 62 datasets, simply run **shortest_62.py**.
5. The visualization of detection results and point-wise metrics are shown in the directory **./eval_demo**. UCR 025 and UCR 150 are used as demo examples, the **test_xxx.txt** contains the Merlin search results, where the columns represent **search_length**, **start_index**, **end_index**, and **discord_distance**. Install the [affiliation metrics](https://github.com/ahstat/affiliation-metrics-py), and run **convert_pw.py**:

    ```
    python convert_pw.py 150
    ```
    which will give the output as:
    
    ```
    Dataset: UCR 150
    window magic correction !!
    UCR 150
    Traditional Metrics:
      F1 Score: 0.3947
    
    PA:
      F1 Score: 0.8619
    
    PA%K - AUC:
      Precision: 0.5859
      Recall: 0.5442
      F1 Score: 0.5466
    
    Affinity:
      Precision: 0.9922
      Recall: 0.9954
    ```

    _*Please note that the experimental outcomes might vary between runs due to the randomness introduced during the augmentation process._

## The TSAD datasets
You can access several widely used TSAD datasets from [Data Smith](https://github.com/pseudo-Skye/Data-Smith/tree/master/TSAD%20Dataset). Additionally, we offer a comprehensive [visualization](https://github.com/pseudo-Skye/Data-Smith/blob/master/TSAD%20Dataset/visualization%20and%20preprocess.ipynb) of them including the UCR dataset. The preprocessed version of the **UCR dataset** utilized in this study is available for direct download [here](https://github.com/pseudo-Skye/Data-Smith/blob/master/TSAD%20Dataset/cleaned_dataset/cleaned_dataset.zip).

## The TSAD evaluation metrics
You may be also interested in this [blog](https://github.com/pseudo-Skye/Time-Matters/blob/main/anomaly%20detection/Anomaly%20in%20TSAD%20Evaluation.md) where we discuss about why the popular evaluation metric, point-adjustment (PA), can be a bit tricky. Additionally, we provide a detailed explanation, along with calculation examples, of the two reliable evaluation metrics used in this study.

