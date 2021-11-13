# SOFT: Softmax-free Transformer with Linear Complexity

![image](resources/structure.png)

> [**SOFT: Softmax-free Transformer with Linear Complexity**](https://arxiv.org/abs/2110.11945),            
> Jiachen Lu, Jinghan Yao, Junge Zhang, Xiatian Zhu, Hang Xu, Weiguo Gao, Chunjing Xu, Tao Xiang, Li Zhang,        
> *NeurIPS 2021 Spotlight* 


## Requirments
* timm==0.3.2

* torch>=1.7.0 and torchvision that matches the PyTorch installation

* cuda>=10.2

Compilation may be fail on cuda < 10.2.  
We have compiled it successfully on `cuda 10.2` and `cuda 11.2`. 

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```
## Installation
```shell script
git clone https://github.com/fudan-zvg/SOFT.git
python -m pip install -e SOFT
```

## Main results
### Image Classification
#### ImageNet-1K

| Model       | Resolution | Params | FLOPs | Top-1 % | Config |Pretrained Model|
|-------------|:----------:|:------:|:-----:|:-------:|--------|--------
| SOFT-Tiny   | 224        | 13M    | 1.9G  | 79.3    |[SOFT_Tiny.yaml](config/SOFT_Tiny.yaml), [SOFT_Tiny_cuda.yaml](config/SOFT_Tiny_cuda.yaml)|[SOFT_Tiny](https://drive.google.com/file/d/1S04DCotIOkP0DaBb8WStQ513z82qT9de/view?usp=sharing), [SOFT_Tiny_cuda](https://drive.google.com/file/d/1inDKh3Wz_2KQgGH_2ywU5H_gLKZpIz_u/view?usp=sharing)
| SOFT-Small  | 224        | 24M    | 3.3G  | 82.2    |[SOFT_Small.yaml](config/SOFT_Small.yaml), [SOFT_Small_cuda.yaml](config/SOFT_Small_cuda.yaml)|
| SOFT-Medium | 224        | 45M    | 7.2G  | 82.9    |[SOFT_Meidum.yaml](config/SOFT_Medium.yaml), [SOFT_Meidum_cuda.yaml](config/SOFT_Medium_cuda.yaml)|
| SOFT-Large  | 224        | 64M    | 11.0G | 83.1    |[SOFT_Large.yaml](config/SOFT_Large.yaml), [SOFT_Large_cuda.yaml](config/SOFT_Large_cuda.yaml)|
| SOFT-Huge   | 224        | 87M    | 16.3G | 83.3    |[SOFT_Huge.yaml](config/SOFT_Huge.yaml), [SOFT_Huge_cuda.yaml](config/SOFT_Huge_cuda.yaml)|

## Get Started

### Train
We have two implementations of Gaussian Kernel: `PyTorch` version and 
the exact form of Gaussian function implemented by `cuda`. The config file containing `cuda` is the 
cuda implementation. Both implementations yield same performance. 
Please **install** SOFT before running the `cuda` version. 
```shell
./dist_train.sh ${GPU_NUM} --data ${DATA_PATH} --config ${CONFIG_FILE}
# For example, train SOFT-Tiny on Imagenet training dataset with 8 GPUs
./dist_train.sh 8 --data ${DATA_PATH} --config config/SOFT_Tiny.yaml
```

### Test

```shell

./dist_train.sh ${GPU_NUM} --data ${DATA_PATH} --config ${CONFIG_FILE} --eval_checkpoint ${CHECKPOINT_FILE} --eval

# For example, test SOFT-Tiny on Imagenet validation dataset with 8 GPUs

./dist_train.sh 8 --data ${DATA_PATH} --config config/SOFT_Tiny.yaml --eval_checkpoint ${CHECKPOINT_FILE} --eval

```
## Reference

```bibtex
@inproceedings{SOFT,
    title={SOFT: Softmax-free Transformer with Linear Complexity}, 
    author={Lu, Jiachen and Yao, Jinghan and Zhang, Junge and Zhu, Xiatian and Xu, Hang and Gao, Weiguo and Xu, Chunjing and Xiang, Tao and Zhang, Li},
    booktitle={NeurIPS},
    year={2021}
}
```

## License

[MIT](LICENSE)


## Acknowledgement

Thanks to previous open-sourced repo:  
[Detectron2](https://github.com/facebookresearch/detectron2)  
[T2T-ViT](https://github.com/yitu-opensource/T2T-ViT)  
[PVT](https://github.com/whai362/PVT)   
[Nystromformer](https://github.com/mlpen/Nystromformer)   
[pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
