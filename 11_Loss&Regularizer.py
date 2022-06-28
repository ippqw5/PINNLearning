# %% [markdown]
# # 损失函数losses以及正则化
#
# 一般来说，监督学习的目标函数由损失函数和正则化项组成。（Objective = Loss + Regularization）
#
# 对于keras模型，目标函数中的正则化项一般在各层中指定，例如使用Dense的 kernel_regularizer 和 bias_regularizer等参数指定权重使用l1或者l2正则化项，此外还可以用kernel_constraint 和 bias_constraint等参数约束权重的取值范围，这也是一种正则化手段。
#
# 损失函数在模型编译时候指定。对于回归模型，通常使用的损失函数是均方损失函数 mean_squared_error。
#
# 对于二分类模型，通常使用的是二元交叉熵损失函数 binary_crossentropy。
#
# 对于多分类模型，如果label是one-hot编码的，则使用类别交叉熵损失函数 categorical_crossentropy。如果label是类别序号编码的，则需要使用稀疏类别交叉熵损失函数 sparse_categorical_crossentropy。
#
# 如果有需要，也可以自定义损失函数，自定义损失函数需要接收两个张量y_true,y_pred作为输入参数，并输出一个标量作为损失函数值。
#
# ## 什么是正则化？
#
# 简单理解：一个额外的loss，称为Regularization。它们在最后与loss相加。<font color='red'> 类似于最优化问题中 **惩罚函数** </font>
#
# **在隐藏层中经常会使用正则来作为损失函数的惩罚项。换言之，为了约束w(bias,output)的可能取值空间从而防止过拟合，我们为该最优化问题加上一个约束，就是w的L1范数或者L2范数不能大于给定值。**
#
# The L1 regularization penalty is computed as:  loss = l1 * reduce_sum(abs(x))
#
# The L2 regularization penalty is computed as: loss = l2 * reduce_sum(square(x))
#
# - kernel_regularizer:对该层中的权值矩阵layer.weights正则
# - bias_regularizer:对该层中的偏差矩阵layer.bias正则
# - activity_regularizer:对该层的输出值矩阵layer.outputs正则
#

# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers,models,losses,regularizers,constraints

# %%
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01), 
                activity_regularizer=regularizers.l1(0.01),
                kernel_constraint = constraints.MaxNorm(max_value=2, axis=0))) 

model.add(layers.Dense(10,
        kernel_regularizer=regularizers.l1_l2(0.01,0.01),activation = "sigmoid"))

model.compile(optimizer = "rmsprop",
        loss = "binary_crossentropy",metrics = ["AUC"])
#"focal_loss"
#"binary_crossentropy"
#FocalLoss
model.summary()
print(model.loss)

# %%
tensor = tf.ones(shape=(1,64),dtype=tf.float32) * 2.0
out = model(tensor)
print(model.input_shape)
print(out)
print(model.losses)


# %% [markdown]
# ### 二，内置损失函数

# %% [markdown]
# 内置的损失函数一般有类的实现和函数的实现两种形式。
#
# 如：CategoricalCrossentropy 和 categorical_crossentropy 都是类别交叉熵损失函数，前者是类的实现形式，后者是函数的实现形式。
#
# 常用的一些内置损失函数说明如下。
#
# * mean_squared_error（均方误差损失，用于回归，简写为 mse, 类与函数实现形式分别为 MeanSquaredError 和 MSE）
#
# * mean_absolute_error (平均绝对值误差损失，用于回归，简写为 mae, 类与函数实现形式分别为 MeanAbsoluteError 和 MAE)
#
# * mean_absolute_percentage_error (平均百分比误差损失，用于回归，简写为 mape, 类与函数实现形式分别为 MeanAbsolutePercentageError 和 MAPE)
#
# * Huber(Huber损失，只有类实现形式，用于回归，介于mse和mae之间，对异常值比较鲁棒，相对mse有一定的优势)
#
# * binary_crossentropy(二元交叉熵，用于二分类，类实现形式为 BinaryCrossentropy)
#
# * categorical_crossentropy(类别交叉熵，用于多分类，要求label为onehot编码，类实现形式为 CategoricalCrossentropy)
#
# * sparse_categorical_crossentropy(稀疏类别交叉熵，用于多分类，要求label为序号编码形式，类实现形式为 SparseCategoricalCrossentropy)
#
# * hinge(合页损失函数，用于二分类，最著名的应用是作为支持向量机SVM的损失函数，类实现形式为 Hinge)
#
# * kld(相对熵损失，也叫KL散度，常用于最大期望算法EM的损失函数，两个概率分布差异的一种信息度量。类与函数实现形式分别为 KLDivergence 或 KLD)
#
# * cosine_similarity(余弦相似度，可用于多分类，类实现形式为 CosineSimilarity)

# %% [markdown]
# ### 三，自定义损失函数

# %% [markdown]
# 自定义损失函数接收两个张量y_true,y_pred作为输入参数，并输出一个标量作为损失函数值。
#
# 也可以对tf.keras.losses.Loss进行子类化，重写call方法实现损失的计算逻辑，从而得到损失函数的类的实现。
#
# 下面是一个Focal Loss的自定义实现示范。Focal Loss是一种对binary_crossentropy的改进损失函数形式。
#
# 它在样本不均衡和存在较多易分类的样本时相比binary_crossentropy具有明显的优势。
#
# 它有两个可调参数，alpha参数和gamma参数。其中alpha参数主要用于衰减负样本的权重，gamma参数主要用于衰减容易训练样本的权重。
#
# 从而让模型更加聚焦在正样本和困难样本上。这就是为什么这个损失函数叫做Focal Loss。
#
# 详见《5分钟理解Focal Loss与GHM——解决样本不平衡利器》
#
# https://zhuanlan.zhihu.com/p/80594704

# %% [markdown]
# $$focal\_loss(y,p) = \begin{cases}
# -\alpha  (1-p)^{\gamma}\log(p) &
# \text{if y = 1}\\
# -(1-\alpha) p^{\gamma}\log(1-p) &
# \text{if y = 0}
# \end{cases} $$

# %%
def focal_loss(gamma=2., alpha=0.75):
    
    def focal_loss_fixed(y_true, y_pred):
        bce = tf.losses.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        loss = tf.reduce_sum(alpha_factor * modulating_factor * bce,axis = -1 )
        return loss
    return focal_loss_fixed



# %%
class FocalLoss(tf.keras.losses.Loss):
    
    def __init__(self,gamma=2.0,alpha=0.75,name = "focal_loss"):
        self.gamma = gamma
        self.alpha = alpha

    def call(self,y_true,y_pred):
        bce = tf.losses.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        loss = tf.reduce_sum(alpha_factor * modulating_factor * bce,axis = -1 )
        return loss
