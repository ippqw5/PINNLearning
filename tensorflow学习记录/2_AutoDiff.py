# -*- coding: utf-8 -*-
# %%
"""

自动微分技术
神经网络依赖反向传播梯度来更新参数，但是求梯度往往是一件复杂且容易出错的事情。
TensorFlow中的 GradientTape——梯度磁带，可以帮我们自动地完成这种工作
tf.GradientTape()首先正向记录运算过程，然后通过反向播放磁带自动得到梯度值。
利用该方法的求梯度过程称为 自动微分技术。
注释：该方法也会在计算图中构建。实际上，任何有关tf的算子（运算）都会被在计算图中构建

"""
import tensorflow as tf
import numpy as np 

x = tf.Variable(1.0,name='x',dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(2.0)
c = tf.constant(3.0)
# %% f(x) = a*x**2 + b*x + c的导数

with tf.GradientTape() as tape:
    f = a*x**2 + b*x + c
df_dx = tape.gradient(f,x)
print(df_dx)

# %% 对常量张量也可以求导，需要增加watch

with tf.GradientTape() as tape:
    tape.watch([a,b,c])
    f = a*x**2 + b*x + c
df_da,df_db,df_dc = tape.gradient(f,[a,b,c])
print(df_da,df_db,df_dc)
# %% 利用嵌套GradientTape 求二阶导数

with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        f = a*x**2 + b*x + c
    df_dx = tape1.gradient(f,x)
df2_dx2 = tape2.gradient(f,x)
print(df2_dx2)
# %% 在autograph(静态图)中使用
@tf.function
def f(x):
    a = tf.constant(1.0)
    b = tf.constant(2.0)
    c = tf.constant(3.0)
    
    # 自变量转换成tf.float32
    x = tf.cast(x,tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = a*x**2 + b*x + c
    return tape.gradient(y,x)
print(f(1.))
# %% 利用梯度磁带和优化器求最小值
# 求f(x) = a*x**2 + b*x + c的最小值
# 使用optimizer.apply_gradients
x = tf.Variable(0.0,name = "x",dtype = tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
for _ in range(0,1000):
    with tf.GradientTape() as tape:
        y = a*x**2 + b*x + c
    dy_dx = tape.gradient(y,x)
    optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])
print("y=",y,"x=",x)

# %%
# 使用optimizer.minimize
# optimizer.minimize相当于先用tape求gradient,再apply_gradient
# 需要定义一个目标函数f，把常量放在函数里面，变量放在函数外面
x = tf.Variable(0.0,name = "x",dtype = tf.float32)
def f():#注意f()无参数
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a*x**2 + b*x + c
    return y
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(0,1000):
    optimizer.minimize(f,[x])
tf.print("y =",f(),"; x =",x)
# %%
# 在autograph中完成最小值求解
# 使用optimizer.apply_gradients
x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
@tf.function
def minimize_auto(t): #注意：手动梯度下降可以传参数，但用optimizer.minimize不行
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    for _ in tf.range(0,1000): #注意autograph时使用tf.range(1000)而不是range(1000)
        with tf.GradientTape() as tape:
            y = a*x**2 + b*x + c
        dy_dx = tape.gradient(y,x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])
    y = a*x**2 + b*x + c
    return y
y = minimize_auto(t=1)
tf.print("y=",y,"x=",x)

# %%
# 在autograph中完成最小值求解
# 使用optimizer.minimize
x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function #这里实际上不用tf.function装饰，只需要在最外层的函数装饰就行了
def f():#注意f()无参数
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a*x**2 + b*x + c
    return y

@tf.function #把函数中的每一个算子都转化成静态图
def train(epoch):
    for _ in tf.range(epoch):
        optimizer.minimize(f,[x])
    return (f())
tf.print("y=",train(1000),"x=",x)

