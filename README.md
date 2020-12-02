# TimeGAN-tensorflow2
## About this repository

Time-series Generative Adversarial Networks(TimeGAN) is the work of Jinsung Yoon, Daniel Jarrett, and Mihaela van der Schaar
([paper](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)). This repository implements TimeGAN ([original code](https://github.com/jsyoon0823/TimeGAN)) with TensorFlow v2.3.1, and mainly for the Hide-and-seek privacy challenge held by NeurIPS ([webpage](https://www.vanderschaar-lab.com/privacy-challenge/)).

## Usage
Execute with default settings (use the Stock dataset)
```
python main.py
```

Execute with differential privacy
```
python main.py --use_dpsgd True
```
