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
import tensorflow_probability as tfp
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


X_u_train = tf.random.normal([1000,2])
u_train = tf.random.normal([1000,1])
X_f_train = tf.random.normal([1000,2])

layer = [2,20,20,20,20,1]
m1= createModel(layer,MyPinn)
m1.compile(keras.optimizers.SGD(learning_rate=0.1))
m1.summary()

m1.train_model(X_u_train,u_train,X_f_train,epochs=5000)


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

# # 7-1 记录
#
# ### Adam优化器 & tfp中L-BFGS

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import numpy as np


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

X_u_train = tf.random.normal([1000,2])
u_train = tf.random.normal([1000,1])
X_f_train = tf.random.normal([1000,2])

layer = [2,20,20,1]
m1= createModel(layer,MyPinn)
m1.compile(keras.optimizers.Adam())
m1.train_model(X_u_train,u_train,X_f_train)


def function_factory(model, loss, X_u_train,u_train,X_f_train):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss(X_u_train,u_train,X_f_train)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f


func = function_factory(m1, m1.loss_Total, X_u_train,u_train,X_f_train)
init_params = tf.dynamic_stitch(func.idx, m1.trainable_variables)

results = tfp.optimizer.lbfgs_minimize(
    value_and_gradients_function=func, initial_position=init_params, max_iterations=5)

m1.loss_Total(X_u_train,u_train,X_f_train)


