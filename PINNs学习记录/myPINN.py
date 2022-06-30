# # 6-29 记录

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def createModel(layer):
    if len(layer) < 2:
        print("层数小于2！, 无法构建神经网络")
        return 
    model = keras.Sequential(name = "PINN")
    model.add(keras.Input(shape=(layer[0],)))
    for i in range(1,len(layer)-1):
        model.add(layers.Dense(layer[i], activation="relu", name="layer{}".format(i)))
    model.add(layers.Dense(layer[-1], name="outputs"))
    
    
    return model

###  测试
layer = [2,4,4,3]
model = createModel(layer)
inputs = tf.random.normal([1,2])
print(model(inputs))
model.summary()


@tf.function
def myloss(y_predict,y_true):
    return tf.reduce_sum(tf.square(y_predict-y_true))


model.compile(
    optimizer=keras.optimizers.SGD(),  # Optimizer
    loss = myloss
    )
model.loss

x = tf.random.normal([10,2])
y = tf.random.normal([10,3])
model.fit(x,y,batch_size=2,epochs=2)

# ## 后续要为model 添加 loss、metric、optimizer
#
#
# model.compile(
#
#     optimizer=keras.optimizers.RMSprop(),  # Optimizer
#     
#     # Loss function to minimize
#     
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     
#     # List of metrics to monitor
#     
#     metrics=[keras.metrics.SparseCategoricalAccuracy()],
#     
# )
#
# > 以上内容截止至 6-29 markdown

# # 6-30 记录

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# train_step()是fit()将会调用的核心函数，
#
# 而train_step()将会使用 我们之前调用compile(optimizer,loss,metrics)时传入的优化器、损失函数、指标。
#
# 实际上compile()做的事情就是将这些部件加入模型，方便train_step()使用。
#
# 故，当我们想要自定义loss、更精准地控制训练过程，同时又想要保留keras.Model或keras.Sequential的方便方法。
#
# 我们可以自定义一个子类，继承keras.Model或keras.Sequential。
#
# 由于PINN的复杂性，需要对预测值求导再组成loss函数，仅仅修改train_step()是不够的，这意味着需要重载fit()，实际上就是抛弃已有的fit()框架，自己定义训练循环。

class MyPinn(keras.Sequential): ## 正在编写
    def __init__(self,name = None):
        
        super(MyPinn, self).__init__(name=name)
        self.nu = tf.constant(0.01/np.pi)
    
    @tf.function
    def test_gradient(self,X_f_train):
        x = X_f_train[:,0]
        t = X_f_train[:,1]
        with tf.GradientTape() as tape:
            tape.watch([x,t])
            X = tf.stack([x,t],axis=-1)
            u = self(X)
        u_x = tape.gradient(u,x)
        tf.print(u_x)
    
    @tf.function
    def loss_U(self,X_u_train,u_train):
        u_pred = self(X_u_train)
        loss_u = tf.reduce_mean(tf.square(u_train - u_pred))
        return loss_u
    
    
    @tf.function
    def loss_PDE(self,X_f_train):
        x = X_f_train[:,0]
        t = X_f_train[:,1]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x,t])
            X = tf.stack([x,t],axis=-1)
            u = self(X)  
            u_x = tape.gradient(u,x)         
            
        u_t = tape.gradient(u, t)     
        u_xx = tape.gradient(u_x, x)
        
        del tape
        
        f = u_t + (self(X_f_train))*(u_x) - (self.nu)*u_xx

        loss_f = tf.reduce_mean(tf.square(f))

        return loss_f
    
    
    def loss_Total(self,X_u_train,u_train,X_f_train):
        loss_u = self.loss_U(X_u_train,u_train)
        loss_f = self.loss_PDE(X_f_train)
        
        loss_total = loss_u + loss_f
        
        return loss_total
    
    @tf.function
    def train_step(self,X_u_train,u_train,X_f_train):
        with tf.GradientTape(persistent=True) as tape:
            loss_total = self.loss_Total(X_u_train,u_train,X_f_train)
                   
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_total, trainable_vars)
        
        del tape
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss_total
    
    def train_model(self, X_u_train,u_train,X_f_train, epochs=100):
        for epoch in tf.range(1,epochs+1):
            loss_total = self.train_step(X_u_train,u_train,X_f_train)
            if epoch % 10 == 0:                
                print(
                    "Training loss (for per 10 epoches) at epoch %d: %.4f"
                    % (epoch, float(loss_total))
                )


def createModel(layer,Model):
    if len(layer) < 2:
        print("层数小于2！, 无法构建神经网络")
        return 
    model = Model(name = "PINN")
    model.add(keras.Input(shape=(layer[0],)))
    for i in range(1,len(layer)-1):
        model.add(layers.Dense(layer[i], activation="relu", name="layer{}".format(i)))
    model.add(layers.Dense(layer[-1], name="outputs"))    
    return model


X_u_train = tf.random.normal([10,2])
u_train = tf.random.normal([10,1])
X_f_train = tf.random.normal([10,2])

layer = [2,4,4,1]
m1= createModel(layer,MyPinn)
m1.compile(keras.optimizers.SGD(learning_rate=0.1))
m1.summary()

m1.train_model(X_u_train,u_train,X_f_train)


## 测试PINN (请无视)
class MyPINN(keras.Sequential):
    

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y_pred = self(x, training=True)  # Forward pass
            loss_u = tf.reduce_mean(tf.square(y-y_pred)) 
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            loss = loss_u

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        del tape
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

# > 以上内容截止至 6-30 markdown
