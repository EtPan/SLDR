## Hyperspectral Image Destriping and Denoising from a Task Decomposition View

### Introduction

This is the source code for our paper: "Hyperspectral Image Destriping and Denoising from a Task Decomposition View".[[url](https://doi.org/10.1016/j.patcog.2023.109832)]

### Usage
#### 1. Requirements

- Python =3.7 
- torch =1.9.0, torchnet, torchvision
- pytorch_wavelets
- pickle, tqdm, tensorboardX, scikit-image

#### 2. Data Preparation

- download ICVL hyperspectral image database from [here](http://icvl.cs.bgu.ac.il/hyperspectral/) 

  save the data in *.mat format into your folder

- generate data with synthetic noise for training and validation

  ```python
     # change the data folder first
      python  ./data/datacreate.py
  ```

- download Real HSI data

  [GF5-baoqing dataset](http://hipag.whu.edu.cn/dataset/Noise-GF5.zip) 

  [GF5-wuhan dataset](http://hipag.whu.edu.cn/dataset/Noise-GF5-2.zip)

#### 3. Training

```python
   python main.py -a phd --dataroot (your data root) --phase train
```

#### 4. Testing

- Testing on Synthetic data with the pre-trained model

  ```python
      python  main.py -a sldr --phase valid  -r -rp checkpoints/model_best.pth
  ```
  
- Testing on Real HSIs with the pre-trained model

  ```python
      python main.py -a sldr --phase test  -r -rp checkpoints/model_best.pth
  ```

### Citation

If you find this work useful, please cite our paper:

```tex
@article{pan2023hyperspectral,
  title={Hyperspectral image destriping and denoising from a task decomposition view},
  author={Pan, Erting and Ma, Yong and Mei, Xiaoguang and Huang, Jun and Chen, Qihai and Ma, Jiayi},
  journal={Pattern Recognition},
  volume={144},
  pages={109832},
  year={2023},
  publisher={Elsevier}
}
```

### Contact 

Feel free to open an issue if you have any question. You could also directly contact us through email at [panerting@whu.edu.cn](mailto:panerting@whu.edu.cn) (Erting Pan)

