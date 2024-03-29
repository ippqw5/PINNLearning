<!-- #region -->

# 学习PINNs and TensorFlow 


记录每日 TensorFlow 学习 & PINN 模型学习
> 参考教材：**《30天吃掉那只TensorFlow2》** https://github.com/lyhue1991/eat_tensorflow2_in_30_days
>
> PINN相关代码 放在 "./PINN学习记录/"中
>
> 详细的每日学习内容记录 放在 "./PINN学习记录/PINNs.md"
>
> 永远祝愿祖国山河统一！



| 日期            | 学习内容                                                     | 备注                                                         |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 8.20-8.21       | Coupled viscous Burgers‘ equation                            | 三个数值算例：两个带解析解，一个无解析解。<br />实验成果/cBurgers_eg1.ipynb,<br />实验成果/cBurgers_eg2.ipynb,<br />实验成果/cBurgers_eg3.ipynb,<br />更新cPINNs报告.md |
| 8.17            | 3D_Parabolic参数反问题                                       | 实验成果/3D_Parabolic区域反问题.ipynb                        |
| 8.15            | 3D_Parabolic区域反问题                                       | 实验成果/3D_Parabolic区域反问题.ipynb                        |
| 8.13-14         | cPINNs报告                                                   | 实验成果/cPINNs报告.md                                       |
| 8.8             | 实验成果                                                     | 实验成果/3D_Parabolic正问题.ipynb                            |
| 8.5             | 各种优化算法学习                                             | 记录 随机梯度下降、牛顿法、动量法、Nesterov、AdaGrad、RMSprop、Adam、Nadam的原理。更新PINN学习记录。 |
| 8.4             | DeepXDE论文阅读(3)                                           | 完结DeepXDE论文，更新PINN学习记录。                          |
| 8.3             | DeepXDE论文阅读(2)                                           | 阅读DeepXDE论文，更新PINN学习记录。                          |
| 8.2             | Effective Tensorflow2(下)<br />DeepXDE论文阅读(1)            | 阅读Effective Tensorflow2文档，阅读DeepXDE论文，看看大佬的优化思路，更新PINN学习记录。 |
| 8.1             | Effective Tensorflow2(上)                                    | 阅读Effective Tensorflow2文档，更新PINN学习记录。            |
| 7.30            | 3d-parabolic耦合pde代码优化                                  | **0730_3D_Parabolic耦合模型_优化.ipynb**<br />添加多种指标Metrics，以便观察训练过程<br />绘图优化 |
| 7.27            | 3d-parabolic耦合pde代码编写                                  | **0727_3D_Parabolic耦合模型.ipynb**<br />(未完工部分：研究3d热力图画法ing) |
| 7.26            | 构造3d的parabolic耦合pde算例<br />DeepXDE库，TensorDiffEq库  | 之前用的2d parabolic耦合pde算例的解析解构造是有规律的，可能拓展到n维。<br />DeepXDE库，TensorDiffEq库为现有的基于PINN等方法求解pde的库。<br />阅读论文：Deep Learning-An Introduction.pdf |
| **7.23**        | 参数反问题                                                   | **0723_参数反问题_Parabolic耦合模型.ipynb**<br />知道部分、全部真解，反推模型参数。 |
| **7.22**        | 区域反问题                                                   | **0722_区域反问题 Parabolic耦合模型.ipynb**<br />简单地说，区域反问题是指边界条件未知，但知道部分内部数值解、真解，反推整个区域的解<br /> |
| 7.21            | 优化训练步骤                                                 | **0721_自适应&LBFGS_Parabolic耦合模型.ipynb**<br />比较Adam算法和LBFGS算法的训练表现。<br />（有必要深入了解Adam的性质，在训练后期表现远不如LBFGS） |
| **7.20**        | 封装耦合PINN代码，使用内置fit<br />自适应因子&预训练模型     | 代码见 **PINN学习记录/7_20_Self_Adaptive_Parabolic耦合pde模型**<br />有关自适应和预训练的说明在**PINN学习记录/PINNs.md** |
| **7.18**        | 更新PINN学习记录&tensorflow学习记录<br />代码 & 论文阅读     | Metric评估函数 & 论文阅读                                    |
| **7.15**        | 改进parabolic耦合pde的代码。                                 | ①耦合训练5000次，loss不怎么下降 ②两个区域分布进行单独的PINN训练 训练准确度提升 |
| 7.13            | 与2位学长会议交流，讨论PINN                                  | 解决了不少疑问，PINN在边界处的拟合效果确实一般。             |
| **7.12**        | 更新PINN学习记录。 训练模型parabolic模型，通过图像，将真实解和PINN模型解做对比。 | u1拟合地很好，u2在多次训练后效果仍然很差。 可能是因为u2中含有y的二次项，而u1中只有y的一次项。 也可能是代码细节出错，正在研究。。。 |
| **7.11**        | 更新PINN学习记录。**使用PINN求解parabolic耦合pde模型**       | 代码见 **7_11_Parabolic耦合pde模型.ipynb** 画图部分还没写，训练数据的生成代码，写的有些冗长，后面会优化一下。 |
| 7.8             | 阅读一篇论文，关于pde耦合模型的数值求解方法。 尝试小批量训练模型，即把训练数据分组，规模减少，但训练次数相应会上升。 |                                                              |
| **7.7**         | 阅读张学长的论文：PINN求解naiver-stokes/Darcy耦合模型的正反问题 | 简单来说，Inputs连接两个独立的NN，分别预测U_{ns} 和 U_{d}， 构造naiver-stokes方程组和Darcy方程组的各自微分方程残差项， 以及interface处的残差项，最后加到一起，形成loss。 |
| **7.6**         | 今天阅读学习了一些NN在PDE求解问题中的多种方法。              |                                                              |
| **7.5**         | 更新PINN学习记录，Debug完成，**代码框架搭建告一段落！**。 后续开始阅读PINN相关论文。 | Debug比写代码要费时间，有一个小错误，看了2个小时硬是没看出来 让我一度以为，搭建思路不行，需要重新搭，或者抄别人。 还有一个“错误”，仅仅仅仅是写法不同，却能导致训练失败。 具体分析在: PINNs学习记录/PINNs.md |
| **7.4**         | 更新PINN学习记录，Data Prep & Plot 代码，调试模型进行训练    | 使用Google colab进行云计算，白嫖算力😍 模型训练没有达到预期，正在debug |
| **7.1**         | 更新PINN学习记录，讨论优化器Adam&L-BFGS，写优化器的代码      | Adam优化器调用起来很简单。 而L-BFGS 需要安装tfp，需要一些复杂的操作才能使用🤣。 |
| **6.30**        | 更新PINN学习记录，完成PINN框架代码搭建。 以**Burgers_Equation**为例，模型训练运行成功，之后将完善细节。 | 今天大概写了一天代码，看了很多官方文档， 本来想调用高阶API训练模型，发现自带的高阶API局限性太强 不如自己重新定义训练过程😢 |
| **6.29**        | 更新PINN学习记录、阅读TensorFlow官方文档                     |                                                              |
| **6.28**        | 11_Loss&Regularizer                                          | 常用损失函数 & 自定义loss & 正则化器                         |
| **6.27**        | 10_tf.layers                                                 | 若干模型层 & 自定义模型层                                    |
| **6.25 - 6.26** | 8_tf.data 数据管道 + 9_tf.Activations 激活函数               | NULL                                                         |
| **~6.25**       | 1_Graph ~ 7_Module                                           | NULL                                                         |



<!-- #endregion -->
