#!/usr/bin/env python
# coding: utf-8
# # tf.Module 上更好地构建 AutoGraph
#

# ### 一，Autograph和tf.Module概述
# 前面在介绍Autograph的编码规范时提到构建Autograph时应该避免在@tf.function修饰的函数内部定义tf.Variable. 
#
# 但是如果在函数外部定义tf.Variable的话，又会显得这个函数有外部变量依赖，封装不够完美。
#
# 一种简单的思路是定义一个类，并将相关的tf.Variable创建放在类的初始化方法中。而将函数的逻辑放在其他方法中。
#
# 这样一顿猛如虎的操作之后，我们会觉得一切都如同人法地地法天天法道道法自然般的自然。
#
# 惊喜的是，TensorFlow提供了一个基类tf.Module，通过继承它构建子类，我们不仅可以获得以上的自然而然，而且可以非常方便地管理变量，还可以非常方便地管理它引用的其它Module，最重要的是，我们能够利用tf.saved_model保存模型并实现跨平台部署使用。
#
# 实际上，tf.keras.models.Model,tf.keras.layers.Layer 都是继承自tf.Module的，提供了方便的变量管理和所引用的子模块管理的功能。
#
# **因此，利用tf.Module提供的封装，再结合TensoFlow丰富的低阶API，实际上我们能够基于TensorFlow开发任意机器学习模型(而非仅仅是神经网络模型)，并实现跨平台部署使用。**
#
#

# ### 二，应用tf.Module封装Autograph

# +
import tensorflow as tf
x = tf.Variable(1.0,dtype=tf.float32)

@tf.function(input_signature=[tf.TensorSpec(shape=[],dtype=tf.float32)])
def add_print(a):
    x.assign_add(a)
    tf.print(x)
    return x

add_print(tf.constant(3.0))


# +
#利用tf.Module 将其封装一下
class DemoModule(tf.Module):
    def __init__(self,init_value = tf.constant(0.0), name = None):
        super(DemoModule,self).__init__(name = name)
        with self.name_scope: #相当于with tf.name_scope("demo_module")
            self.x = tf.Variable(init_value,dtype = tf.float32, trainable = True)
    @tf.function(input_signature=[tf.TensorSpec(shape=[],dtype=tf.float32)])
    def addprint(self,a):
        with self.name_scope:
            self.x.assign_add(a)
            tf.print(self.x)
            return self.x

demo = DemoModule(init_value = tf.constant(1.0))
demo.addprint(tf.constant(5.0))
# -

print(demo.variables)
print(demo.trainable_variables)

demo.submodules

#使用tf.saved_model 保存模型，并指定需要跨平台部署的方法
tf.saved_model.save(demo,"./data/demo/1",signatures = {"serving_default":demo.addprint})

#加载模型
demo2 = tf.saved_model.load("./data/demo/1")
demo2.addprint(tf.constant(5.0))

# 查看模型文件相关信息，红框标出来的输出信息在模型部署和跨平台使用时有可能会用到
# !saved_model_cli show --dir ./data/demo/1 --all

# +
#在tensorboard中查看计算图，模块会被添加模块名demo_module,方便层次化呈现计算图结构
import datetime

# 创建日志
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = './data/demomodule/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

#开启autograph跟踪
tf.summary.trace_on(graph=True, profiler=True) 

#执行autograph
demo = DemoModule(init_value = tf.constant(0.0))
result = demo.addprint(tf.constant(5.0))

#将计算图信息写入日志
with writer.as_default():
    tf.summary.trace_export(
        name="demomodule",
        step=0,
        profiler_outdir=logdir)
# -

#启动 tensorboard在jupyter中的魔法命令
# %reload_ext tensorboard

from tensorboard import notebook
notebook.list() 

notebook.start("--logdir ./data/demomodule/")

# 除了利用tf.Module的子类化实现封装，我们也可以通过给tf.Module添加属性的方法进行封装。

# +
mymodule = tf.Module()
mymodule.x = tf.Variable(0.0)
@tf.function(input_signature = [tf.TensorSpec(shape=[], dtype = tf.float32)])
def addprint(a):
    mymodule.x.assign_add(a)
    tf.print(mymodule.x)
    return mymodule.x

mymodule.addprint = addprint
# -

mymodule.addprint(tf.constant(1.0)).numpy()

print(mymodule.variables)

# ### 三，tf.Module和tf.keras.Model，tf.keras.layers.Layer
# tf.keras中的模型和层都是继承tf.Module实现的，也具有变量管理和子模块管理功能。

import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics


print(issubclass(tf.keras.Model,tf.Module))
print(issubclass(tf.keras.layers.Layer,tf.Module))
print(issubclass(tf.keras.Model,tf.keras.layers.Layer))

# +
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(4,input_shape=(10,)))
model.add(layers.Dense(2))
model.add(layers.Dense(1))
model.summary()
# -

model.variables

model.layers[0].trainable = False #冻结第0层的变量，使其不可训练
model.trainable_variables

model.submodules

model.layers

print(model.name)
print(model.name_scope())
