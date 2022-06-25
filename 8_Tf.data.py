# %% [markdown]
# # 5-1,数据管道Dataset
#
# 如果需要训练的数据大小不大，例如不到1G，那么可以直接全部读入内存中进行训练，这样一般效率最高。
#
# 但如果需要训练的数据很大，例如超过10G，无法一次载入内存，那么通常需要在训练的过程中分批逐渐读入。
#
# 使用 tf.data API 可以构建数据输入管道，轻松处理大量的数据，不同的数据格式，以及不同的数据转换。

# %% [markdown]
# # 一、构建数据管道

# %%
import tensorflow as tf
import numpy as np
from sklearn import datasets

# %%
dataset = tf.data.Dataset.from_tensor_slices([8.0,2.2,1.1,5.0])
dataset

# %%
for elem in dataset:
    print(elem.numpy())

# %%
it = iter(dataset)
print(next(it).numpy)

# %%
print(dataset.reduce(0.,lambda state,value: state + value).numpy())

# %% [markdown]
# ### <font color = 'green'>Dataset.element_spec 返回中每一个元素的定义</font>

# %%
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4,10]))
dataset1.element_spec

# %%
dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([4]),
    tf.random.uniform([4,100], maxval = 100, dtype=tf.int32)))

dataset2.element_spec

# %% [markdown]
# ### <font color = 'green'>通过zip把两个dataset中的元素 组成一个新的元组后构成一个dataset</font>

# %%
dataset3 = tf.data.Dataset.zip((dataset1,dataset2))
dataset3.element_spec


# %%
dataset3

# %%
for a,(b,c) in dataset3:
    print("shapes: {a.shape},{b.shape},{c.shape}".format(a=a,b=b,c=c))

# %% [markdown]
#

# %%
train, test =  tf.keras.datasets.fashion_mnist.load_data()

# %%
type(train)

# %%
images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images,labels))
dataset
# x是28x28的,y是标量

# %%
type(images)

# %% [markdown]
# ###  <font color = 'green'>1.从numpy 构建数据管道</font>

# %%
import tensorflow as tf
import numpy as np
from sklearn import datasets

# %%
iris = datasets.load_iris()
iris.keys()

# %%
ds1 = tf.data.Dataset.from_tensor_slices( (iris['data'],iris['target']) )
for feature,label in ds1.take(5):
    tf.print(feature, label)

# %%

# %% [markdown]
# ### <font color='green'> 2.从Pandas构建数据管道</font>

# %%
import tensorflow as tf
from sklearn import datasets 
import pandas as pd

iris = datasets.load_iris()
d_iris = pd.DataFrame( iris['data'], columns=iris['feature_names'])
ds2 = tf.data.Dataset.from_tensor_slices((d_iris.to_dict("list"),iris["target"]))

for feature,label in ds2.take(3):
    tf.print(feature,label)

# %% [markdown]
# ### <font color='green'> 3.从Python generator 构建数据管道</font>

# %% [markdown]
# 在 Python 中，使用了 yield 的函数被称为生成器（generator）。
#
# 跟普通函数不同的是，生成器是一个返回迭代器的函数，只能用于迭代操作，更简单点理解生成器就是一个迭代器。
#
# 在调用生成器运行的过程中，每次遇到 yield 时函数会暂停并保存当前所有的运行信息，返回 yield 的值, 并在下一次执行 next() 方法时从当前位置继续运行。
#
# 调用一个生成器函数，返回的是一个迭代器对象。

# %%
# 从Python generator构建数据管道
import tensorflow as tf

def count(stop):
    i = 0
    while (i<stop):
        yield i
        i += 1

for n in count(5):
    print(n)

# %%
ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )
for count_batch in ds_counter.repeat().batch(10).take(5):
    print(count_batch.numpy())

# %%
ds_counter.repeat().batch(10)


# %% [markdown]
# The output_shapes argument is not required but is highly recommended as many TensorFlow operations do not support tensors with an unknown rank. If the length of a particular axis is unknown or variable, set it as None in the output_shapes.
#
# It's also important to note that the output_shapes and output_types follow the same nesting rules as other dataset methods.
#
# Here is an example generator that demonstrates both aspects: it returns tuples of arrays, where the second array is a vector with unknown length.

# %%
def gen_series():
  i = 0
  while True:
    size = np.random.randint(0, 10)
    yield i, np.random.normal(size=(size,))
    i += 1



# %%
for i, series in gen_series():
  print(i, ":", str(series))
  if i > 5:
    break

# %% [markdown]
# The first output is an int32 the second is a float32.
#
# The first item is a scalar, shape (), and the second is a vector of unknown length, shape (None,)

# %%
ds_series = tf.data.Dataset.from_generator(
            gen_series,
            output_types=(tf.int32,tf.float32),
            output_shapes=( (), (None,) ))

ds_series

# %% [markdown]
# Now it can be used like a regular tf.data.Dataset. 
#
# Note that when batching a dataset with a variable shape, you need to use Dataset.padded_batch.

# %%
ds_series_batch = ds_series.shuffle(20).padded_batch(10)

ids, sequence_batch = next(iter(ds_series_batch))
print(ids.numpy())
print()
print(sequence_batch.numpy())

# %%

# %% [markdown]
# # 二、应用数据转换

# %% [markdown]
# Dataset数据结构应用非常灵活，因为它本质上是一个Sequece序列，其每个元素可以是各种类型，例如可以是张量，列表，字典，也可以是Dataset。
#
# Dataset包含了非常丰富的数据转换功能。
#
# * map: 将转换函数映射到数据集每一个元素。
#
# * flat_map: 将转换函数映射到数据集的每一个元素，并将嵌套的Dataset压平。
#
# * interleave: 效果类似flat_map,但可以将不同来源的数据夹在一起。
#
# * filter: 过滤掉某些元素。
#
# * zip: 将两个长度相同的Dataset横向铰合。
#
# * concatenate: 将两个Dataset纵向连接。
#
# * reduce: 执行归并操作。
#
# * batch : 构建批次，每次放一个批次。比原始数据增加一个维度。 其逆操作为unbatch。
#
# * padded_batch: 构建批次，类似batch, 但可以填充到相同的形状。
#
# * window :构建滑动窗口，返回Dataset of Dataset.
#
# * shuffle: 数据顺序洗牌。
#
# * repeat: 重复数据若干次，不带参数时，重复无数次。
#
# * shard: 采样，从某个位置开始隔固定距离采样一个元素。
#
# * take: 采样，从开始位置取前几个元素。
#

# %%
# map： 将转换函数 映射到数据集的每一个元素上

ds = tf.data.Dataset.from_tensor_slices(["hello world","hello china","hello shanghai"])
ds_map = ds.map(lambda x: tf.strings.split(x," "))
for x in ds_map:
    print(x)

# %%
# flat_map:将转换函数映射到数据集的每一个元素，并将嵌套的Dataset压平。
ds = tf.data.Dataset.from_tensor_slices(["hello world","hello china","hello shanghai"])
ds_flatmap = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x," ")))
for x in ds_flatmap:
    print(x)

# %%
# interleave
ds = tf.data.Dataset.from_tensor_slices(["hello world","hello china","hello shanghai"])
ds_interleave = ds.interleave(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x," ")))
for x in ds_interleave:
    print(x)

# %%
