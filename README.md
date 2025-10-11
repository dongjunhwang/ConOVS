# OVS Meets Continual Learning: Towards Sustainable Open-Vocabulary Segmentation (NeurIPS 2025)

[[Paper (NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/115181)] [Paper (arXiv)]

by [Dongjun Hwang](https://dongjunhwang.github.io/), [Yejin Kim](https://sites.google.com/view/yejin-c-kim/), [Minyoung Lee](https://sites.google.com/view/minyoung-lee), [Seong Joon Oh](https://coallaoh.github.io/), [Junsuk Choe](https://sites.google.com/site/junsukchoe/)

This repo contains the code for our paper **"OVS Meets Continual Learning: Towards Sustainable Open-Vocabulary Segmentation"**.

This code is based on [**fc-clip**](https://github.com/bytedance/fc-clip) and [**mask2former**](https://github.com/facebookresearch/Mask2Former). Many thanks to all for sharing the excellent code!


<br>

<div align="center">
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/39bdebba-9353-4571-8ea0-3af1b2591d9b">
</div>

<br>

> **Abstract**: Open-Vocabulary Segmentation (OVS) aims to segment classes that are not present in the training dataset. However, most existing studies assume that the training data is fixed in advance, overlooking more practical scenarios where new datasets are continuously collected over time. To address this, we first analyze how existing OVS models perform under such conditions. In this context, we explore several approaches such as retraining, fine-tuning, and continual learning but find that each of them has clear limitations. To address these issues, we propose ConOVS, a novel continual learning method based on a Mixture-of-Experts framework. ConOVS dynamically combines expert decoders based on the probability that an input sample belongs to the distribution of each incremental dataset. Through extensive experiments, we show that ConOVS consistently outperforms existing methods across pre-training, incremental, and zero-shot test datasets, effectively expanding the recognition capabilities of OVS models when data is collected sequentially.

---

### 🔥 TODO
- [ ] Release the gpaper on HuggingFace
- [ ] Upload the paper to arXiv
- [X] Update code  

---
## Preparation
### Installation
See [installation instructions](INSTALL.md).

### Preparing Datasets
See [Preparing Datasets](datasets/README.md).

### Preparing a Pre-trained Model

Please run the below command and put that on the root directory of this repo.
```bash
gdown 1csp3trVKhc90aUZO4S__V23iwsQTe0od
```
And move this file to `./checkpoints/fcclip_cocopan.pth`.

### Preparing the Evaluation

You can evaluate all the below method without training the model as we already provide the checkpoint [[download link](https://drive.google.com/drive/folders/1kzRaPB_BVh37a9dTYVE66ueRfIg--Fbt?usp=sharing)].

Downloaded files should be in the project directory like below:
```
ConOVS/
├── ...
├── checkpoints/
│   ├── fcclip_cocopan.pth
│   ├── finetuning/
│   │   ├── cityscapes.pth
│   ...
├── mvn_dist/
│   ├── cityscapes_gmm.pkl
│   ...
```


**Notes:**
- **Fine-tuning**: Name of the file indicates the sequence of the incremental datasets. 
- **Retraining**: Name of the file indicates all incremental datasets.
- **ConOVS**: You should download the checkpoint from `./finetuning` in the link above, and MVN distribution from [here](https://drive.google.com/drive/folders/1fWsYEORvCax0lVc4txu5yKUBHnQXugAg?usp=drive_link).

---

## Getting Started with ConOVS

This document provides a brief intro of the usage of our method and other comparisons.

This document is derived from  [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md).

### Process of Our Method
Our method consists of two main steps: 
- (1) fine-tuning the fc-clip model on the first dataset, and 
- (2) making the MVN distribution (prototypes) to combine the fine-tuned models.

**We already provided the fine-tuned models and the prototypes in [this link](https://drive.google.com/drive/folders/1yX3GvjVa8fZp8ewTBclJ27zGMzXh77-j?usp=drive_link).**

Therefore, if you want to evaluate our method, you can skip the steps (1) and (2) and directly go to the [Evaluation](#Evaluation) section.

### (1) Fine-tuning the fc-clip model

We provide a script `train_net.py`, that is made to train all the configs.

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](./datasets/README.md),
then beginning with fine-tuning fc-clip:
```bash
# scripts/(scenario number)/(method name).sh
bash scripts/s1/finetuning.sh
```
Other methods can be trained in a similar manner. For example, to train the retraining model, use
```bash
bash scripts/s1/retraining.sh
```

### (2) Generating the Multivariate Normal Distribution

To generate the multivariate normal distribution,
```
bash scripts/s1/conovs_gmm.sh
```

## Evaluation

To evaluate ConOVS's performance, use
```
bash scripts/s1/conovs_eval.sh
```
Other methods can be evaluated in a similar manner. For example, to evaluate the fine-tuned model, use
```
bash scripts/s1/finetuning_eval.sh
```
For more options, see `python train_net.py -h`.

## Citation

TBD