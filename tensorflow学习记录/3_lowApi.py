# -*- coding: utf-8 -*-
# %%
"""
低阶API 示范
主要包括： 张量操作、计算图和自动微分
"""
# %%
# 打印时间分割线
import tensorflow as tf

@tf.function
def printbar():
    today_s = tf.timestamp() % (24*60*60)
    
    hour = tf.cast(today_s//3600 + 8,tf.int32)
    minute = tf.cast((today_s%3600)//60,tf.int32)
    second = tf.cast((today_s%3600)%60,tf.int32)
    
    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m)) == 1:
            return tf.strings.format("0{}",m)
        else:
            return tf.strings.format("{}",m)
    timestring = tf.strings.join([timeformat(hour),timeformat(minute),timeformat(second)],
                                 separator = ":")
    tf.print("======="*8+timestring)
printbar()
# %%
tf.gather([1,2,3,4],[0,1])

# %% 一、线性回归模型
#准备模型 (测试用) 单独运行
import numpy as py
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

#样本数
n = 400

#生成测试数据集
X = tf.random.uniform([n,2],minval=-10,maxval=10)
w0 = tf.constant([[2.0],[-3.0]])
b0 = tf.constant([[3.0]])
Y = X@w0 + b0 + tf.random.normal([n,1],mean=0.0,stddev=2.0) #@是矩阵乘法，增加正态噪音

#
w = tf.Variable([[1.5],[1.5]])
b = tf.Variable([[1.0]])

def model(x):
    return x@w + b
def loss(y_true,y_predict):
    return tf.reduce_mean((y_true-y_predict)**2/2)

def train(optimizer,x,y_true):
    with tf.GradientTape() as tape:
        y_predict = model(x) 
        l = loss(y_true,y_predict)
    dl_dw,dl_db = tape.gradient(l,[w,b])
    optimizer.apply_gradients(grads_and_vars=[(dl_dw,w),(dl_db,b)])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
@tf.function
def train_loop():    
    for i in tf.range(400):
        x = tf.reshape(X[i],(1,2))
        y_true = x@w0 + b0 
        train(optimizer,x,y_true)  
    tf.print("w= ",w,"\nb=",b)  
# %% 一、线性回归模型
#准备模型
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

#样本数
n = 400

#生成测试数据集
X = tf.random.uniform([n,2],minval=-10,maxval=10)
w0 = tf.constant([[2.0],[-3.0]])
b0 = tf.constant([[3.0]])
Y = X@w0 + b0 + tf.random.normal([n,1],mean=0.0,stddev=2.0) #@是矩阵乘法，增加正态噪音


# %%
# 数据分组 By batch_size ，返回一个管道迭代器
def data_iter(features,labels,batch_size):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices) #随机打乱顺序
    for i in range(0,num_examples,batch_size):
        indexs = indices[i: min(i+batch_size,num_examples)]
        yield tf.gather(features,indexs), tf.gather(labels,indexs) 
batch_size = 8
Container = data_iter(X,Y,batch_size)
(features,labels) = next(Container)
print("----- batch_1 -----",)
print("features:",features)
print("labels:",labels)
(features,labels) = next(Container)
print("----- batch_2 -----",)
print("features:",features)
print("labels:",labels)

# %%
# 2. 定义模型
w = tf.Variable(tf.random.normal(w0.shape))
b = tf.Variable(tf.zeros_like(b0,dtype=tf.float32))

class LinearRegression:
    #正向传播
    def __call__(self,x):
        return x@w + b
    #损失函数
    def loss_func(self,y_true,y_predict):
        return tf.reduce_mean((y_true-y_predict)**2/2)

model = LinearRegression()


# %%
# 3.训练模型

# 动态图 单步训练模型
def train_loop(model,feature,label):
    with tf.GradientTape() as tape:
        y_predict = model(feature)
        loss = model.loss_func(label,y_predict)
    #反向传播求梯度
    dloss_dw,dloss_db = tape.gradient(loss,[w,b])
    #梯度下降法更新梯度
    w.assign_sub(0.01 * dloss_dw)
    b.assign_sub(0.01 * dloss_db)
    
    return loss



# %%
# 测试训练效果
batch_size = 10
(feature,label) = next(data_iter(X,Y,batch_size))
train_loop(model,feature,label)


# %%
def train_model(model,epochs):
    for epoch in tf.range(0,epochs+1):
        for (features,labels) in data_iter(X,Y,10):
            loss = train_loop(model,features,labels)
        if epoch%50 == 0:
            printbar()
            tf.print("epoch=",epoch,"loss=",loss)
            tf.print("w=",w)
            tf.print("b=",b)
epochs = 200
train_model(model,epochs)


# %%
# 使用autograph机制转换为静态图急速
@tf.function
def train_loop(model,feature,label):
    with tf.GradientTape() as tape:
        y_predict = model(feature)
        loss = model.loss_func(label,y_predict)
    #反向传播求梯度
    dloss_dw,dloss_db = tape.gradient(loss,[w,b])
    #梯度下降法更新梯度
    w.assign_sub(0.01 * dloss_dw)
    b.assign_sub(0.01 * dloss_db)
    
    return loss
def train_model(model,epochs):
    for epoch in tf.range(0,epochs+1):
        for (features,labels) in data_iter(X,Y,10):
            loss = train_loop(model,features,labels)
        if epoch%50 == 0:
            printbar()
            tf.print("epoch=",epoch,"loss=",loss)
            tf.print("w=",w)
            tf.print("b=",b)
epochs = 200
train_model(model,epochs)

# %%
# 二 DNN二分类模型
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import tensorflow as tf
#正负样本数量
n_positive = 2000
n_negative = 2000

#生成正样本，小圆环分布
r_p = 5.0 + tf.random.truncated_normal([n_positive,1],0.0,1.0)
theta_p = tf.random.uniform([n_positive,1],0.0,2*np.pi)
Xp = tf.concat([r_p*tf.cos(theta_p),r_p*tf.sin(theta_p)], axis=1)
Yp = tf.ones_like(r_p) # r_p.shape ones-Tensor

#生成负样本，大圆环分布
r_n = 10.0 + tf.random.truncated_normal([n_negative,1],0.0,1.0)
theta_n = tf.random.uniform([n_negative,1],0.0,2*np.pi)
Xn = tf.concat([r_n * tf.cos(theta_n), r_n * tf.sin(theta_n)], axis=1)
Yn = tf.zeros_like(r_n) 

#样本汇总
X = tf.concat([Xp,Xn], axis=0)
Y = tf.concat([Yp,Yn], axis=0)

#可视化
plt.figure(figsize = (6,6))
plt.scatter(Xp[:,0].numpy(),Xp[:,1].numpy(),c="r")
plt.scatter(Xn[:,0].numpy(),Xn[:,1].numpy(),c="b")


# %%
# 数据分组 By batch_size ，返回一个管道迭代器
def data_iter(features,labels,batch_size):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices) #随机打乱顺序
    for i in range(0,num_examples,batch_size):
        indexs = indices[i: min(i+batch_size,num_examples)]
        yield tf.gather(features,indexs), tf.gather(labels,indexs) 
        
batch_size = 10
(features,labels) = next(data_iter(X,Y,batch_size))
print("features:",features)
print("labels:",labels)


# %%
# 2.定义模型
# 此处范例利用tf.Module 来组织模型变量

class DNNModel(tf.Module):
    def __init__(self,name = None):
        super(DNNModel, self).__init__(name=name)
        self.w1 = tf.Variable(tf.random.truncated_normal([2,4]), dtype = tf.float32)
        self.b1 = tf.Variable(tf.zeros([1,4]), dtype=tf.float32)
        self.w2 = tf.Variable(tf.random.truncated_normal([4,8]), dtype = tf.float32)
        self.b2 = tf.Variable(tf.zeros([1,8]), dtype=tf.float32)
        self.w3 = tf.Variable(tf.random.truncated_normal([8,1]), dtype = tf.float32)
        self.b3 = tf.Variable(tf.zeros([1,1]), dtype = tf.float32)
        
    #正向传播
    @tf.function(input_signature=[tf.TensorSpec(shape = [None,2], dtype=tf.float32)])
    def __call__(self,x):
        x = tf.nn.relu(x@self.w1 + self.b1)
        x = tf.nn.relu(x@self.w2 + self.b2)
        y = tf.nn.sigmoid(x@self.w3 + self.b3)
        return y
        
    # 损失函数(二元交叉熵)
    @tf.function(input_signature=[tf.TensorSpec(shape = [None,1], dtype = tf.float32),
                              tf.TensorSpec(shape = [None,1], dtype = tf.float32)])  
    def loss_func(self,y_true,y_pred):  
        #将预测值限制在 1e-7 以上, 1 - 1e-7 以下，避免log(0)错误
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred,eps,1.0-eps)
        bce = - y_true*tf.math.log(y_pred) - (1-y_true)*tf.math.log(1-y_pred)
        return  tf.reduce_mean(bce)
    
    # 评估指标（正确率）
    @tf.function(input_signature=[tf.TensorSpec(shape = [None,1], dtype = tf.float32),
                                 tf.TensorSpec(shape = [None,1], dtype = tf.float32)])
    def metric_func(self,y_true,y_pred):
        y_pred = tf.where( y_pred > 0.5, tf.ones_like(y_pred , dtype = tf.float32),
                         tf.zeros_like(y_pred, dtype = tf.float32))
        acc = tf.reduce_mean(1 - tf.abs( y_true - y_pred))
        return acc
    
model = DNNModel()

# %%
# 测试模型结构
batch_size = 10

(features,labels) = next(data_iter(X,Y,batch_size))

predictions = model(features)

loss = model.loss_func(labels,predictions)
metric = model.metric_func(labels,predictions)

tf.print("init loss:",loss)
tf.print("init metric:",metric)

# %%
print(len(model.trainable_variables))
model.trainable_variables


# %%
# 3.训练模型
# 使用autograph机制转换成静态图加速

@tf.function
def train_step(model,features,labels):
    
    #正向传播求损失
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func(labels,predictions)
    #反向传播求梯度
    grads = tape.gradient(loss, model.trainable_variables)
    
    #梯度下降
    for p, dloss_dp in zip(model.trainable_variables, grads):
        p.assign_sub( 0.001 * dloss_dp)
    
    #计算评估指标
    metric = model.metric_func(labels,predictions)
    
    return loss,metric

def train_model(model,epochs):
    for epoch in tf.range(1,epochs+1):
        for features, labels in data_iter(X,Y,100):
            loss, metric = train_step(model,features,labels)
        if epoch%100 == 0:
            printbar()
            tf.print("epoch = ",epoch,"loss = ",loss,"accuracy = ", metric)

train_model(model, epochs=600)

# %%
# 结果可视化
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize = (12,5))
ax1.scatter(Xp[:,0],Xp[:,1],c = "r")
ax1.scatter(Xn[:,0],Xn[:,1],c = "g")
ax1.legend(["positive","negative"]);
ax1.set_title("y_true");

Xp_pred = tf.boolean_mask(X,tf.squeeze(model(X)>=0.5),axis = 0)
Xn_pred = tf.boolean_mask(X,tf.squeeze(model(X)<0.5),axis = 0)

ax2.scatter(Xp_pred[:,0],Xp_pred[:,1],c = "r")
ax2.scatter(Xn_pred[:,0],Xn_pred[:,1],c = "g")
ax2.legend(["positive","negative"]);
ax2.set_title("y_pred");
