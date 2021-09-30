# PhyCRNet

Physics-informed convolutional-recurrent neural networks for solving spatiotemporal PDEs 

Paper link: [[ArXiv](https://arxiv.org/pdf/2106.14103.pdf)]

By: [Pu Ren](https://scholar.google.com/citations?user=7FxlSHEAAAAJ&hl=en), [Chengping Rao](https://github.com/Raocp), [Yang Liu](https://coe.northeastern.edu/people/liu-yang/), [Jian-Xun Wang](http://sites.nd.edu/jianxun-wang/) and [Hao Sun](https://web.mit.edu/haosun/www/#/home)

## Highlights
- Present a Physics-informed discrete learning framework for solving spatiotemporal PDEs without any labeled data
- Proposed an encoder-decoder convolutional-recurrent scheme for low-dimensional feature extraction
- Employ hard-encoding of intial and boundary conditions
- Incorporate autoregressive and residual connections to explicitly simulate the time marching

We demonstrate excellent solution accuracy, extrapolability and generalizability on three nonlinear PDEs as follows.



1. Traing and Extrapolation 

2D Burgers' equaions, $\Lambda$-$\Omega$ reaction diffusion equations, and FitzHuge-Nagumo reaction diffusion eqautions.

2. Generalization

FitzHuge-Nagumo reaction diffusion eqautions


insert figure here ...



## Introduction
todo

## Code and datasets
todo

## Citation
If you find our research helpful, please consider citing us with：

```
@article{ren2021phycrnet,
  title={PhyCRNet: Physics-informed Convolutional-Recurrent Network for Solving Spatiotemporal PDEs},
  author={Ren, Pu and Rao, Chengping and Liu, Yang and Wang, Jianxun and Sun, Hao},
  journal={arXiv preprint arXiv:2106.14103},
  year={2021}
}
```
