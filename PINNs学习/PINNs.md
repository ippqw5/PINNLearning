

# 耦合PINN、正反问题、3D问题的学习研究



## 记录：2022-06-29 

### 什么是PINN？

Raissi等人在2018年 

 [Physics-informed neural networks: A deep learning 
framework for solving forward and inverse problems involving 
nonlinear partial differential equations](https://github.com/maziarraissi/PINNs)

中提出了PINN，通过在损耗函数中结合物理场（即偏微分方程）和边界条件来解决偏微分方程。损失是偏微分方程的均方误差和在分布在域中的“搭配点”上测量的边界残差。



他们在文章中列出了数个经典pde算例来验证PINN的正确性。在他们的github主页上提供了源码，使用了Tensorflow1.x的框架。

截止到2022-06-29，TensorFlow更新到2.9，TensorFlow1.x代码风格跟TensorFlow2.x相差甚远，而且Raissi等人提供的代码也有许多值得优化的地方。

我在github上找到一个了PINN在TensorFlow2.x上的代码框架 [omniscientoctopus/Physics-Informed-Neural-Networks: Investigating PINNs (github.com)](https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks),



### 本文研究什么？

利用暑假研究一下如下问题： 

- PINN模型在耦合问题上的应用（这方面的研究似乎还不多），

- PINN正负问题求解（看了一些文章说PINN的优势其实在于 **高维问题 & 反问题求解**
- 代码实现若干算例（也许可能maybe Only one😁），最好能把模型应用到3维。

> 以上内容于 2022 -  06 - 29  markdown。

