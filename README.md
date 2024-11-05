# Trend and Order Features for Semi-supervised Time Series Classification via Multi-task Learning

UniTS is a unified time series model that can process various tasks across multiple domains with shared parameters and does not have any task-specific modules.

Authors: [Rongjun chen](https://scholar.google.cz/citations?hl=zh-CN&user=8O_9j3EAAAAJ), Xuanhui Yan, [Guobao Xiao](https://scholar.google.cz/citations?hl=zh-CN&user=YC2B2OEAAAAJ), Yi Hou, Shilin Zhou

## Overview
 Multitask learning with pretext task has excelled in time series classification task lacking labeled data. The key to multitask learning is to build a pretext task and learn the most representative feature from raw time series. In this paper, we propose Trend and Order Features for semi-supervised time series classification via multitask Learning (TOFL). Specially, we propose a simple but effective pretext task --- self-sequence order prediction (SOP) --- to  discover the order relation.In addition, we design a Gradual Trend Fusion block concatenating different trend features as the shared backbone network basis element to obtain high quality trend features for SOP task. Finally, we not only theoretically analyze the uniform stability and generalization error of TOFL, but also evaluate the results compared with state-of-the-art (SOTA) supervised and semi-supervised methods on the 128 UCR datasets and three real-world datasets.  TOFL demonstrates a high level of competitiveness and, in most cases, closely matches or even surpasses SOTA methods in terms of accuracy.

<p align="center">
<img src="./img/SOP.png" height = "190" alt="" align=center />
</p>




## Setups

### 1. Requirements
 Install Pytorch2.0+ and the required packages.
```
pip install -r requirements.txt
```

### 2. Prepare data
You can access the well pre-processed datasets from [[Baidu Drive\]](https://pan.baidu.com/s/1SXyuErkML3sylmvDKn3aaA?pwd=krvm), then place the downloaded contents under `./datasets`

### 3. Quick Demos

You can run the command as follow:

```
python main.py --dataset_name CricketX --model_name TOFL
```



## Acknowledgement

This codebase is built based on the [SemiTime](https://github.com/haoyfan/SemiTime). Thanks!

yfan/SemiTime). Thanks!

