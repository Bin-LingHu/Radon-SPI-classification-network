⚡Title:
Radon Single-Pixel Flying Target Classification via Texture-Fused Lightweight Differentiable Operators
Hubin Ling, Bingzhang Hu, Dongfeng Shi, Yinbo Huang, and Yingjian Wang


⚡Requirements
PyTorch and timm 0.6.11 (`pip install timm==0.6.11`).


⚡DataSet
 # Radon Single Pixel Flying Object Classification Dataset

     dataset
       |----train_GT
               |------00 aircraft
               |------01 bird
               |------03 UAV
       |----train_FBP2
               |------00 aircraft
               |------01 bird
               |------03 UAV
       |----train_FBP5
               |------00 aircraft
               |------01 bird
               |------03 UAV   
       |----train_FBP1
               |------00 aircraft
               |------01 bird
               |------03 UAV  
       |----test
               |------00 aircraft
               |------01 bird
               |------03 UAV
       |----val
               |------00 aircraft
               |------01 bird
               |------03 UAV
       |----...


⚡Codes
## Validation

To evaluate models, using following examples:
Path, model name, pretrained model can be modified according to your files.

```bash
   python validate.py ../dataset/test/test_GT --model mambaout_ltpe --checkpoint ./output/train/mamba_ltpe_224V6_GT/model_best.pth.tar
```

## Train
To train models, using following examples:
Path, model name, pretrained model can be modified according to your files.

```bash
python train.py ../dataset/ --train-split train_FBP2 --val-split ./test/test_FBP2 --model mambaout_ltpe -b 16 --opt adamw --lr 1e-3 --epochs 300
```


⚡Info
## Bibtex
```
@article{RadonSPIFly2025,
  author = {Ling, Hubin and Hu, Bingzhang and Shi, Dongfeng and Huang, Yinbo and  Wang, Yingjian},
  title = {Radon Single-Pixel Flying Target Classification via Texture-Fused Lightweight Differentiable Operators},
  journal = {Scientific Reports},
  year = {2025},
  pages = {1-15},
}
```


## Acknowledgment 

Our implementation is based on [MambaOut-main](https://github.com/yuweihao/MambaOut), [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), [poolformer](https://github.com/sail-sg/poolformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), [metaformer](https://github.com/sail-sg/metaformer), and [inceptionnext](https://github.com/sail-sg/inceptionnext).









