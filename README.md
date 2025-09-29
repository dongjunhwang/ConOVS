# OVS Meets Continual Learning: Towards Sustainable Open-Vocabulary Segmentation (NeurIPS 2025)
[[Paper (NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/115181)] [Paper (arXiv)]

by [Dongjun Hwang](https://dongjunhwang.github.io/), [Yejin Kim](https://sites.google.com/view/yejin-c-kim/), [Minyoung Lee](https://sites.google.com/view/minyoung-lee), [Seong Joon Oh](https://coallaoh.github.io/), [Junsuk Choe](https://sites.google.com/site/junsukchoe/)

<br>

<div align="center">
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/72472a63-399f-40d2-8c73-b499caaa04aa">
</div>

<br>

> **Abstract**: Open-Vocabulary Segmentation (OVS) aims to segment classes that are not present in the training dataset. However, most existing studies assume that the training data is fixed in advance, overlooking more practical scenarios where new datasets are continuously collected over time. To address this, we first analyze how existing OVS models perform under such conditions. In this context, we explore several approaches such as retraining, fine-tuning, and continual learning but find that each of them has clear limitations. To address these issues, we propose ConOVS, a novel continual learning method based on a Mixture-of-Experts framework. ConOVS dynamically combines expert decoders based on the probability that an input sample belongs to the distribution of each incremental dataset. Through extensive experiments, we show that ConOVS consistently outperforms existing methods across pre-training, incremental, and zero-shot test datasets, effectively expanding the recognition capabilities of OVS models when data is collected sequentially.

---

### 🔥 TODO
- [ ] Release the paper on HuggingFace  
- [ ] Update code  
- [ ] Upload the paper to arXiv
