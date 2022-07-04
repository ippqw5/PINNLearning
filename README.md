<!-- #region -->
# 学习PINNs and TensorFlow 


记录每日 TensorFlow 学习 & PINN 模型学习
> 参考教材：**《30天吃掉那只TensorFlow2》** https://github.com/lyhue1991/eat_tensorflow2_in_30_days



| 日期            | 学习内容                                                     | 备注                                                         |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **~6.25**       | 1_Graph ~ 7_Module                                           | NULL                                                         |
| **6.25 - 6.26** | 8_tf.data 数据管道 + 9_tf.Activations 激活函数               | NULL                                                         |
| **6.27**        | 10_tf.layers                                                 | 若干模型层 & 自定义模型层                                    |
| **6.28**        | 11_Loss&Regularizer                                          | 常用损失函数 & 自定义loss & 正则化器                         |
| **6.29**        | 更新PINN学习记录、阅读TensorFlow官方文档                     |                                                              |
| **6.30**        | 更新PINN学习记录，完成PINN框架代码搭建。<br />以**Burgers_Equation**为例，模型训练运行成功，之后将完善细节。<br /> | 今天大概写了一天代码，看了很多官方文档，<br />本来想调用高阶API训练模型，发现自带的高阶API局限性太强<br />不如自己重新定义训练过程😢 |
| **7.1**         | 更新PINN学习记录，讨论优化器Adam&L-BFGS，写优化器的代码      | Adam优化器调用起来很简单。<br />而L-BFGS 需要安装tfp，需要一些复杂的操作才能使用🤣。 |
| 7.4             | 更新PINN学习记录，Data Prep & Plot 代码，调试模型进行训练    | 使用Google colab进行云计算，白嫖算力😍<br />模型训练没有达到预期，正在debug |





<!-- #endregion -->
