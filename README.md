# Reproducing [Inception Transformer](http://arxiv.org/abs/2205.12956)
This is a Non-official PyTorch implementation of iFormer proposed by paper "[Inception Transformer](http://arxiv.org/abs/2205.12956)".

**I suppose there are some details missing in my implementation.**
**Parameters and FLOPs are 90% alike**

# iFormer in The PAPER

| Model      |  #params  | FLOPs | Image resolution |
| :---       |   :---:   |  :---: |  :---: |
| iFormer-S  |   20M     |   4.8G  |   224 |
| iFormer-B  |   48M     |   9.4G  |   224 |
| iFormer-L  |   87M     |   14.0G |   224 |

# iFormer Reproduce —— Profile by THOP
| Model      |  #params  | FLOPs | Image resolution |
| :---       |   :---:   |  :---: |  :---: |
| iFormer-S  |   18.45M     |   4.29G  |   224 |
| iFormer-B  |   46.96M     |   8.59G  |   224 |
| iFormer-L  |   87.33M     |   13.25G |   224 |