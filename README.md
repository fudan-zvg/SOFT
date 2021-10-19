# SOFT: Softmax-free Transformer with Linear Complexity

![image](resources/structure.png)

> [**SOFT: Softmax-free Transformer with LinearComplexity**](https://arxiv.org/abs/2012.15840),            
> Jiachen Lu, Jinghan Yao, Junge Zhang, Xiatian Zhu, Hang Xu, Weiguo Gao, Chunjing Xu, Tao Xiang, Li Zhang,        
> *NeurIPS 2021* 


## Requirments
timm

torch>=1.7.0 and torchvision that matches the PyTorch installation

cuda>=10.2

## Installation
```shell script
https://github.com/fudan-zvg/SOFT.git
python -m pip install -e SOFT
```

## Main results
### Image Classification
#### ImageNet-1K

| Model       | Resolution | Params | FLOPs | Top-1 % |
|-------------|:----------:|:------:|:-----:|:-------:|
| SOFT-Tiny   | 224        | 13M    | 1.9G  | 79.3    |
| SOFT-Small  | 224        | 24M    | 3.3G  | 82.2    |
| SOFT-Medium | 224        | 25M    | 7.2G  | 82.9    |
| SOFT-Large  | 224        | 64M    | 11.0G | 83.1    |
| SOFT-Huge   | 224        | 87M    | 16.3G | 83.3    |

## Get Started

### Train

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
# For example, train a SETR-PUP on Cityscapes dataset with 8 GPUs
./tools/dist_train.sh configs/SETR/SETR_PUP_768x768_40k_cityscapes_bs_8.py 8
```

### Test

```shell
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}  [--eval ${EVAL_METRICS}]
# For example, test a SETR-PUP on Cityscapes dataset with 8 GPUs
./tools/dist_test.sh configs/SETR/SETR_PUP_768x768_40k_cityscapes_bs_8.py \
work_dirs/SETR_PUP_768x768_40k_cityscapes_bs_8/iter_40000.pth \
8 --eval mIoU
```
## Reference

```bibtex
@inproceedings{SOFT,
    title={SOFT: Softmax-free Transformer with LinearComplexity}, 
    author={Lu, Jiachen and Yao, Jinghan and Zhang, Junge and Zhu, Xiatian and Xu, Hang and Gao, Weiguo and Xu, Chunjing and Xiang, Tao and Zhang, Li},
    booktitle={NeurIPS},
    year={2021}
}
```

## License

MIT


## Acknowledgement

Thanks to previous open-sourced repo:  
[Detectron2](https://github.com/facebookresearch/detectron2)     
[T2T-ViT](https://github.com/yitu-opensource/T2T-ViT)  

