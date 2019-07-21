# SimplePruning
This repository provides a cnn channels pruning demo with tensorflow. You can pruning your own model(support conv2d,depthwise conv2d,pool,fc,concat ops...) defined in modelsets.py. Have a good time!

&nbsp;[![author Haibo](https://img.shields.io/badge/author-Haibo%20Wong-blue.svg?style=flat)](https://github.com/DasudaRunner/Object-Tracking)&nbsp;&nbsp;&nbsp;&nbsp;
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)<br>
- &emsp;***Author**: Haibo Wang*<br>
- &emsp;***Email**: dasuda2015@163.com*
- &emsp;***Home Page**: [dasuda.top](https://dasuda.top)*

---
## Dependencies

- `Tensorflow >= 1.10.0`
- `python >= 3.5`
- `opencv-python >= 4.1.0`
- `numpy >= 1.14.5`
- `matplotlib >= 3.0.3`

---
## Getting Started

- ##### Clone the repository
```bash
  $ git clone https://github.com/DasudaRunner/SimplePruning.git
```

- ##### Downdload the Cifar10 dataset, and put into cifar-10-python/
  Url: `http://www.cs.toronto.edu/~kriz/cifar.html`

- ##### (**Optional**) Define your model in modesets.py

  `You must use add_layer() API defined in pruner.py to set up your model. More details to modelsets.py`

- ##### (**Optional**) Config params in utils/config.py
  `e.g. model name, learning rate, pruning rate.`

- ##### Train a full model, .ckpt and .pb model file will be saved in ckpt_model/
```bash
  $ python full_train.py
```

- ##### Channel pruning. .ckpt and .pb model file will be saved in channels_pruned_model/
```bash
  $ python channels_pruning.py
```

---
## Evaluation on Cifar10 dataset

| Model | Dataset | Pruning rate | Model size / MB | Inference time / ms\*64pic |
|:-:|:-:|:-:|:-:|:-:|
|SimpleNet|cifar-10| 0.5 |8.7 -> 1.8| 5.8 -> 2.7|
|VGG19|cifar-10 | 0.5 |53.4 -> 13.5|28.62 -> 9.44|
|DenseNet40|cifar-10| 0.5 |4.3 -> 1.5|77.87 -> 39.97|
|MobileNet V1|cifar-10| 0.5 |6.6 -> 1.8|19.39 -> 8.01|
|OCR-Net|---|0.5|2426.2 -> 841.9|10.36->7.3|