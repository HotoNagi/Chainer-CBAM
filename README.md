# Chainer-CBAM
# Chainer-CBAM(Convolutional Block Attention Module)

## Training
```
python n_fold_train.py --dataset [DATASET] --arch [MODEL] --gpu [GPU_ID]
```
## 軽い解説
[論文リンク](https://arxiv.org/abs/1807.06521)  
### 使用ネットワーク
ResBlockに以下の図のような **Channel Attention Module** , **Spatial Attention Module** と呼ばれる2つの Attention Module を追加したもの。  
![](https://i.imgur.com/ZRTgAV3.png)  
![](https://i.imgur.com/35HZtbB.png)  
![](https://i.imgur.com/khy6Bze.png)  
![](https://i.imgur.com/KvOBTJm.png)  

### Channel Attention Module
![](https://i.imgur.com/de36HVg.png)  

### Spatial Attention Module
![](https://i.imgur.com/v4SY9NB.png)  

## 参考実装
* [公式（pytorch）](https://github.com/luuuyi/CBAM.PyTorch)  
* [非公式（pytorch）](https://github.com/Jongchan/attention-module)
