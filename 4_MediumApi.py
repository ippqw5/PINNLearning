"""
中级 API

各种模型层，损失函数，优化器，数据管道，特征列等待

"""

# +
import tensorflow as tf

#打印时间分割线
@tf.function
def printbar():
    today_ts = tf.timestamp()%(24*60*60)

    hour = tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)
    minite = tf.cast((today_ts%3600)//60,tf.int32)
    second = tf.cast(tf.floor(today_ts%60),tf.int32)
    
    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m))==1:
            return(tf.strings.format("0{}",m))
        else:
            return(tf.strings.format("{}",m))
    
    timestring = tf.strings.join([timeformat(hour),timeformat(minite),
                timeformat(second)],separator = ":")
    tf.print("=="*8+timestring)


# -

printbar()

# +
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
import tensorflow as tf
from tensorflow.keras import layers,losses,metrics,optimizers

#样本数量
n = 400

# 生成测试用数据集
X = tf.random.uniform([n,2],minval=-10,maxval=10) 
w0 = tf.constant([[2.0],[-3.0]])
b0 = tf.constant([[3.0]])
Y = X@w0 + b0 + tf.random.normal([n,1],mean = 0.0,stddev= 2.0)  # @表示矩阵乘法,增加正态扰动

# -

#构建输入数据管道
ds = tf.data.Dataset.from_tensor_slices((X,Y)) \
     .shuffle(buffer_size = 100).batch(10) \
     .prefetch(tf.data.experimental.AUTOTUNE)  

# 2、定义模型
model = layers.Dense(units = 1) #继承 Layer 和 Module
model.build(input_shape = (2,)) # 用build方程方法创建variables
model.loss_func = losses.mean_squared_error # MSE
model.optimizer = optimizers.SGD(learning_rate = 0.001)


# +
# 3. 训练模型

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func( tf.reshape(labels,[-1]), tf.reshape(predictions,[-1] ))
    grads = tape.gradient(loss,model.variables)
    model.optimizer.apply_gradients(zip(grads, model.variables))
    
    return loss
 
features, labebls = next(ds.as_numpy_iterator())
train_step(model,features,labels)
# -

ds.as_numpy_iterator()


def train_loop(model,epochs):
    for epoch in tf.range(1,epochs+1):
        loss = tf.constant(0.0)
        for features, labels in ds:
            loss = train_step(model,features,labels)
        if epoch%50 == 0:
            printbar()
            tf.print("epoch=",epoch,"loss=",loss)
            tf.print("w=",model.variables[0])
            tf.print("b=",model.variables[1])
train_loop(model,200)

# +
# 二、DNN
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers,losses,metrics,optimizers
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

#正负样本数量
n_positive,n_negative = 2000,2000

#生成正样本, 小圆环分布
r_p = 5.0 + tf.random.truncated_normal([n_positive,1],0.0,1.0)
theta_p = tf.random.uniform([n_positive,1],0.0,2*np.pi) 
Xp = tf.concat([r_p*tf.cos(theta_p),r_p*tf.sin(theta_p)],axis = 1)
Yp = tf.ones_like(r_p)

#生成负样本, 大圆环分布
r_n = 8.0 + tf.random.truncated_normal([n_negative,1],0.0,1.0)
theta_n = tf.random.uniform([n_negative,1],0.0,2*np.pi) 
Xn = tf.concat([r_n*tf.cos(theta_n),r_n*tf.sin(theta_n)],axis = 1)
Yn = tf.zeros_like(r_n)

#汇总样本
X = tf.concat([Xp,Xn],axis = 0)
Y = tf.concat([Yp,Yn],axis = 0)


#可视化
plt.figure(figsize = (6,6))
plt.scatter(Xp[:,0].numpy(),Xp[:,1].numpy(),c = "r")
plt.scatter(Xn[:,0].numpy(),Xn[:,1].numpy(),c = "g")
plt.legend(["positive","negative"]);
# -

ds = tf.data.Dataset.from_tensor_slices( (X,Y) ) \
        .shuffle(buffer_size = 4000).batch(100) \
        .prefetch(tf.data.experimental.AUTOTUNE)

# +
# 定义模型



class DNNModel(tf.Module):
    def __init__(self, name = None):
        super(DNNModel, self).__init__(name = name)
        self.dense1 = layers.Dense(4,activation = "relu")
        self.dense2 = layers.Dense(8,activation = "relu")
        self.dense3 = layers.Dense(1,activation = "sigmoid")
        
    #正向传播
    @tf.function(input_signature = [ tf.TensorSpec( shape = [None,2], dtype=tf.float32)])
    def __call__(self,x):
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.dense3(x)
        return y
model = DNNModel()
model.loss_func = losses.binary_crossentropy
model.metric_func = metrics.binary_accuracy
model.optimizer = optimizers.Adam(learning_rate = 0.001)

# +
#测试
features,labels = next(ds.as_numpy_iterator())
predictions = model(features)

loss = model.loss_func( tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]) )
metric = model.metric_func( tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]) )

tf.print("init loss:",loss)
tf.print("init metric:",metric)


# -

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func( tf.reshape(labels,[-1]), tf.reshape(predictions,[-1] ))
    grads = tape.gradient(loss,model.variables)
    model.optimizer.apply_gradients(zip(grads, model.variables))
    
    metric = model.metric_func( tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]) )

    
    return loss,metric
# 测试
features , labels = next(ds.as_numpy_iterator())
train_step(model,features,labels)


def train_model(model,epochs):
    for epoch in tf.range(1,epochs+1):
        loss = tf.constant(0.0)
        metric = tf.constant(0.0)
        for features, labels in ds:
            loss, metric = train_step(model,features,labels)
        if epoch%10 == 0:
            printbar()
            tf.print("epoch = ",epoch,"loss = ",loss,"accuracy = ",metric)
train_model(model,60)
