# %% [markdown]
# # 计算图 
#      对于张量的运算都是需要构建计算图的。过程中，并没有显示地构造和展示出计算图，不过其计算路径确实是沿着计算图的路径来进行的。
#      
# ## Tensorflow 2.x  Eager模式 (默认）
#      在计算图的构建过程中，同时输出计算结果。—— 动态图
#      Eager模式方便调试，运行效率慢些，特别对于需要反复调用的代码段。
#     
#      注：在TensorFlow1.x 中，推崇***静态图***的形式。运行效率高。
#      即，任何的张量计算都必须完整地构建计算图，这个过程不会输出计算结果。在计算图构建完成后，将输入“喂给”计算图，开始执行计算。
# ## Tensorflow 2.x AutoGraph模式
#      AutoGraph 取动态图和静态图之长，既可以方便调试，又可以提升运算速度。特别对于需要反复调用的代码段。
#      
#      编写规范：
#      @tf.function
#      def cal_tensor():
#          c =  a + b
#          return c
#      
# #     使用AutoGraph能够帮助我们观察到计算图构建的过程。
#
#
# %%
import tensorflow as tf
import numpy as np


# %%
# b = x*y  + z
def diff():
    x = tf.constant(1.0)
    y = tf.constant(2.0)
    z = tf.constant(3.0)
    
    with tf.GradientTape() as tape:
        tape.watch([x,y,z])
        a = x * y
        b = a + z
    db_x,db_y,db_z = tape.gradient(target=[b],sources=[x,y,z])
    tf.print("db_dx=",db_x,",db_dy=",db_y,",db_dz=",db_z)
diff()

# %%
@tf.function
def add():
    a = tf.constant(1.0)
    b = tf.constant(1.0)
    c = a + b
    tf.print(c)
    print("tracing")
    


# 后面什么都没有发生。仅仅是在Python堆栈中记录了这样一个函数的签名。
#
# **当我们第一次调用这个被@tf.function装饰的函数时，后面到底发生了什么？
#
# 例如我们写下如下代码。
# %%
add()
add()
