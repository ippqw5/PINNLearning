# -*- coding: utf-8 -*-
"""
Created on Sun May 22 00:39:14 2022

tensorflow2.x 计算图: 动态图（default） and autoGraph（静态图）

@author: ippqw
"""
import tensorflow as tf
#%%
"""
动态图(default)
计算图在每个算子（运算符号）处都进行构建、构建完后立即执行 
"""
x = tf.constant("Hello")
y = tf.constant("World")
z = tf.strings.join([x,y],separator=' ')
tf.print(z)
print(z)

# 将动态计算图代码的输入和输出关系封装成函数

def strjoin(x,y):
    z = tf.strings.join([x,y],separator=' ')
    tf.print(z)
    return z
result = strjoin(tf.constant("hello"), tf.constant("world"))

#%%
"""
AutoGraph 静态图； 运行效率比动态图快一些
必须要以函数的形式封装，并用@tf.function装饰器
"""
@tf.function
def strjoin(x,y):
    z = tf.strings.join([x,y],separator=' ')
    tf.print(z)
    return z
result = strjoin(tf.constant("hello"), tf.constant("world"))
print(result)
#%%
