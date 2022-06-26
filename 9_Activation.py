# %% [markdown]
# # 5-3,激活函数activation
#
# 激活函数在深度学习中扮演着非常重要的角色，它给网络赋予了非线性，从而使得神经网络能够拟合任意复杂的函数。
#
# 如果没有激活函数，无论多复杂的网络，都等价于单一的线性变换，无法对非线性函数进行拟合。
#
# tf.keras.activations

# %% [markdown]
# # <font color='green'> 一.常用激活函数 </font>

# %%
import tensorflow as tf

# %% [markdown]
# **1.tanh**
#
# tf.keras.activations.tanh(x)
#
# Tensor of same shape and dtype of input x, with tanh activation: $tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x))).$
#
# 将实数压缩到-1到1之间，输出期望为0。主要缺陷为存在梯度消失问题，计算复杂度高。

# %%
a = tf.constant([-3.0,1.0,2.0],dtype=tf.float32)
b = tf.keras.activations.tanh(a)
print(b)

# %% [markdown]
# **2.Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)).**
#
# tf.keras.activation.sigmoid(x)
#
# 常用于最后一层，做二分类问题，因为0-1之间刚好描述概率
#
# 当x<-5,sigmod(x) = 0, x>5,sigmod(x)=1

# %%
a = tf.constant([-20.0, -1.0, 0.0, 1.0, 20.0],dtype=tf.float32)
b = tf.keras.activations.tanh(a)
print(b)

# %% [markdown]
# **3. Relu activation function, Relu(x) = max(x,0)**
#
# tf.keras.activations.relu(
#     x, alpha=0.0, max_value=None, threshold=0.0
# )
#
# alpha: 当x小于0时，不是直接归零，而是 乘以 alpha
#
# max_value: 正数的最大值
#
# threshold: 分界点，将函数横向平移

# %%
a = tf.constant([-20.0, -1.0, 0.0, 1.0, 20.0],dtype=tf.float32)
b = tf.keras.activations.relu(a)
c = tf.keras.activations.relu(a,alpha=0.5,max_value=5.0,threshold=0.0)
print(b)
print(c)

# %% [markdown]
# **4. hard-sigmoid**
# A faster approximation of the sigmoid activation. Piecewise linear approximation of the sigmoid function.
#
# The hard sigmoid activation, defined as:
# - if x < -2.5: return 0
# - if x > 2.5: return 1
# - if -2.5 <= x <= 2.5: return 0.2 * x + 0.5

# %%
a = tf.constant([-20.0, -1.0, 0.0, 1.0, 20.0],dtype=tf.float32)
b = tf.keras.activations.hard_sigmoid(a)
print(b)

# %% [markdown]
# **5. elu: Exponential Linear Unit.**
#
# alpha > 0 is: x if x > 0 and alpha * (exp(x) - 1) if x < 0 

# %%
a = tf.constant([-20.0, -1.0, 0.0, 1.0, 20.0],dtype=tf.float32)
b = tf.keras.activations.elu(a)
print(b)

# %% [markdown]
# **6. gelu: Gaussian error linear unit(GELU)**
#
# tf.keras.activations.gelu(
#     x, approximate=False
# )
#
# The gaussian error linear activation: 
#
# 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))) if approximate is True 
#
# or x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2))), where P(X) ~ N(0, 1), if approximate is False.

# %%
a = tf.constant([-20.0, -1.0, 0.0, 1.0, 20.0],dtype=tf.float32)
b = tf.keras.activations.gelu(a)
c = tf.keras.activations.gelu(a,approximate=True)
print(b)
print(c)

# %% [markdown]
# **7. softmax :处理多分类问题**
#
# 将向量处理成概率分布的形式
#
# tf.keras.activations.softmax(
#     x, axis=-1
# )
#
# The softmax of each vector x is computed as exp(x) / tf.reduce_sum(exp(x)).

# %%
inputs = tf.random.normal( shape=(32,10) ) #默认最后一维看成特征维度，这里是10，即每个向量的长度是10。而32可以看做是batch_size
outputs = tf.keras.activations.softmax( inputs )

print(tf.reduce_sum( outputs[0,:]))
