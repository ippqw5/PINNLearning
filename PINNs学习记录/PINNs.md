py

# 耦合PINN、正反问题、3D问题的学习研究



## 记录时间：2022-06-29 

### 什么是PINN？

Raissi等人在2018年 

 [Physics-informed neural networks: A deep learning 
framework for solving forward and inverse problems involving 
nonlinear partial differential equations](https://github.com/maziarraissi/PINNs)

中提出了PINN，通过在损耗函数中结合物理场（即偏微分方程）和边界条件来解决偏微分方程。损失是偏微分方程的均方误差和在分布在域中的“搭配点”上测量的边界残差。



他们在文章中列出了数个经典pde算例来验证PINN的正确性。在他们的github主页上提供了源码，使用了Tensorflow1.x的框架。

截止到2022-06-29，TensorFlow更新到2.9，TensorFlow1.x代码风格跟TensorFlow2.x相差甚远，而且Raissi等人提供的代码也有许多值得优化的地方。

我在github上找到一个了PINN在TensorFlow2.x上的代码框架 [omniscientoctopus/Physics-Informed-Neural-Networks: Investigating PINNs (github.com)](https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks),



### 本文研究什么？

利用暑假研究一下如下问题： 

- PINN模型在耦合问题上的应用（这方面的研究似乎还不多），
- PINN正负问题求解（看了一些文章说PINN的优势其实在于 **高维问题 & 反问题求解**
- 代码实现若干算例（也许可能maybe Only one😁），最好能把模型应用到3维。



### PINN模型搭建

> 工欲善其事必先利其器，先掌握TensorFlow，并使用TensorFlow自定义PINN模型是必要的。

先设想一个代码框架。

PINN的模型比较简单，如果先不考虑loss函数，那么PINN就是一个普通的全连接序列模型，可以用 **tf.keras.Sequential()**构建。

假设Layer = [inputs,n1,n2,...nk,n_outputs]，从左至右代表每层神经元的个数。

```python
def createModel(layer):
    if len(layer) < 2:
        print("层数小于2！, 无法构建神经网络")
        return 
    model = keras.Sequential(name = "PINN")
    model.add(keras.Input(shape=(layer[0],)))
    for i in range(1,len(layer)-1):
        model.add(layers.Dense(layer[i], activation="relu", name="layer{}".format(i)))
    model.add(layers.Dense(layer[-1], name="outputs"))
    
    #  model.compile( loss= , metrics = [], optimizer= )
    #	...
    #
    #
    
    return model
```



#### <font color='purple'>后续要为 model 添加 loss（PINN主要部分）、metric（可以不要）、optimizer（优化器，必要）</font>

> 以上内容于 2022 -  06 - 29  markdown。

---

## 记录时间：2022-06-30

### <font color='blue'>1.PINN模型搭建（续）</font>

TensorFlow为我们提供了多种多样的高阶API帮助我们快速搭建模型。**但是，快速方便的代价就是灵活性降低。**特别是PINN跟一般的机器学习模型训练步骤不同，主要来源于loss function需要对预测值进行求导运算。

今天的代码编写遭遇了一些困难，我昨天的构思实际上不可行：

1. 使用 **tf.keras.Sequential()**或者**tf.keras.Model()函数式API** 搭建模型 Model。(这一步没问题）

2. 编写自定义loss函数，然后将 自定义的loss & 优化器 & metrics 传入自带的方法 **Model.compile()**

3. 调用自带的方法 **Model.fit()** ，传入训练数据训练即可。



步骤1 和 （步骤2，步骤3）是相对独立的。而PINN模型结构简单，使用步骤1搭建模型完全没有问题。

问题出在（步骤2，步骤3）。

让我们先来看看其中两个关键的方法 **Model.compile() & Model.fit()** 方法究竟做了什么？

如果明白了Model.fit()的工作步骤，就会知道这种自带的训练方式局限性，以及为什么在训练PINN模型时，我们需要自定义训练过程。



**Model.fit()大致工作流程如下：**

```python
def fit(self,X_train,Y_train,epoches,batch_size,**kwargs):
    #
    #		主要的训练部分代码
    #
	for epoch in epoches: ## 使用 X_train,Y_train 进行 epoch 次训练
        for x_batch,y_batch in dataset(X_train,Y_train,batch_size): ## dataset(X_train,Y_train,batch_size) 数据分组
            train_step((x_batch,y_batch))  ## train_step(self,data) 是 类中的方法，可以重载，从而改变fit()

def train_step(self,data):
    
    # Unpack the data. Its structure depends on your model and on what you pass to `fit()`.
    x, y = data

    with tf.GradientTape(persistent=True) as tape:
        y_pred = self(x, training=True)  # Forward pass
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        #这里用到的 self.compiled_loss() 就是 Model.compile(loss) 传入的loss
        #即 Model.compile() 作用是让 self.compiled_loss = loss
                       
	# Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    del tape

    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    
    #指标计算（可选）
    #....
```



可以看到 **Model.fit()**只是为我们定义一个一般普通的训练步骤函数，并没有什么特殊。

**Model.fit()** 会在 train_step() 中调用 Model.compile() 传入的loss、优化器、指标。

```python
loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
```

上述代码中compiled_loss()限定了参数 y , y_pred。y是标签值，y_pred是通过模型预测的值。



这也是为什么官方文档、网上的帖子说：

**如果想自定义loss，那么你自定义的loss函数一定要有两个输入参数（y，y_pred)。**

**因为在 fit() ——》train_step()中，调用了self.compiled_loss()，并规定了它的参入参数为（y,y_pred)**。



当然，我们可以重载 train_step() 使自定义loss函数自由度更高，并仍能使用fit()。

甚至重载fit()，但既然都重载fit()了，它等价于从0编写训练过程。——不过，这正是PINN训练需要的。

**因为我们有不同格式的训练数据（带标签和不带标签）、复杂的loss函数（不仅仅需要y，y_pred，还需要y_pred对x，t的导数），所以在这种情况下，仍然使用fit（）代码框架，需要大刀阔斧地修改。还不如自己写训练过程，不去使用所谓的高阶API~~**



### <font color='blue'>2.子类化Sequential() / Model() , 定义 MyPinn，自定义训练过程</font>

之前讨论过**步骤1：使用Sequential搭建模型** 是没有问题的，我们需要做的仅仅是重新定义**训练过程**。

那么，子类化**class MyPinn(tf.keras.Sequential())**，MyPinn保留Sequential()好用的功能，再附加上我们自定义的**训练过程**

```python
class MyPinn(keras.Sequential): ## 以Burgers_Equation为例
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
```



<font color='purple'> **在当前文件夹 myPINN.py 进行了train_model测试，成功运行**</font>

> 以上内容截止至 6-30 markdown
