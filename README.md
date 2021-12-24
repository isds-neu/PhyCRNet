# PhyCRNet

Physics-informed convolutional-recurrent neural networks for solving spatiotemporal PDEs 

Paper link: [[Journal Paper](https://www.sciencedirect.com/science/article/pii/S0045782521006514)], [[ArXiv](https://arxiv.org/pdf/2106.14103.pdf)]

By: [Pu Ren](https://scholar.google.com/citations?user=7FxlSHEAAAAJ&hl=en), [Chengping Rao](https://github.com/Raocp), [Yang Liu](https://coe.northeastern.edu/people/liu-yang/), [Jian-Xun Wang](http://sites.nd.edu/jianxun-wang/) and [Hao Sun](https://web.mit.edu/haosun/www/#/home)

## Highlights
- Present a Physics-informed discrete learning framework for solving spatiotemporal PDEs without any labeled data
- Proposed an encoder-decoder convolutional-recurrent scheme for low-dimensional feature extraction
- Employ hard-encoding of initial and boundary conditions
- Incorporate autoregressive and residual connections to explicitly simulate the time marching



### Training and Extrapolation

Here we show the comparison between PhyCRNet and PINN on 2D Burgers' equations below. The left, middle and right figures are the ground truth, the result from our PhyCRNet and the result from PINNs respectively.

<p align="center">
  <img src="https://user-images.githubusercontent.com/55661641/135552658-c3c2c955-dc12-4995-8451-d3f524af1405.gif" width="512">
</p>



### Generalization

We show the generalization test on FitzHugh-Nagumo reaction-diffusion equations with four different initial conditions. The left and right parts are the ground truth generated with the high-order finite difference method and the results from our PhyCRNet, respectively.

<p align="center">
  <img src="https://user-images.githubusercontent.com/55661641/135554104-ef5ee5dd-a707-4448-9634-89b23a4c8858.gif" width="200">
  <img src="https://user-images.githubusercontent.com/55661641/135554152-ab0d830e-e2eb-489e-8faf-8b9298072a36.gif" width="200">
  <img src="https://user-images.githubusercontent.com/55661641/135554156-efd65c12-2ab2-4ceb-bb3e-719cdf636710.gif" width="200">
  <img src="https://user-images.githubusercontent.com/55661641/135554165-1d4f9d41-795f-4d4d-b7fa-0299b2c45fca.gif" width="200">
</p>



## Requirements

- Python 3.6.13
- [Pytorch](https://pytorch.org/) 1.6.0
- Other packages, such as *Matplotlib, Numpy and Scipy*, are also used

## Datasets

We provide the codes for data generation used in this paper, including 2D Burgers' equations and 2D FitzHugh-Nagumo reaction-diffusion equations. They are coded in the high-order finite difference method. Besides, the code for random field is modified from [[Link](https://github.com/zongyi-li/fourier_neural_operator/tree/master/data_generation/navier_stokes)]. You may find the data solver for λ-ω reaction-diffusion equations in [[Link](https://github.com/snagcliffs/PDE-FIND/tree/master/Datasets)]. 

The initial conditions tested in this paper are also provided in the folder **Datasets**.

## Codes

The general code of PhyCRNet is provided in the folder **Codes**, where we use 2D Burgers' equations as a testing example. For other PDE systems, the network setting is similar. You may try modifying the grid sizes and time steps to your own cases.

It is important to pretrain the network from a small number of time steps (e.g., 100) and then gradually train on a longer dynamical evolution. Taking 2D Burgers' equation as an example, we pretrain the model from 100, then 200 and 500, and finally 1000. The model is able to extrapolate for at least another 1000 time steps.

## Citation
If you find our research helpful, please consider citing us with：

```
@article{ren2022phycrnet,
  title={PhyCRNet: Physics-informed convolutional-recurrent network for solving spatiotemporal PDEs},
  author={Ren, Pu and Rao, Chengping and Liu, Yang and Wang, Jian-Xun and Sun, Hao},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={389},
  pages={114399},
  year={2022},
  publisher={Elsevier}
}
```
