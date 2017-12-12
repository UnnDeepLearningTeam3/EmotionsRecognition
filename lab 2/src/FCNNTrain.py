import mxnet as mx
import logging

logging.getLogger().setLevel(logging.DEBUG)

"""Train data consist of 5 characters with 7 different emotions we need to recognize"""
train_data = mx.io.ImageRecordIter(
  path_imgrec="./data/diff_data_train.rec", # The target record file.
  data_shape=(3, 128, 128), #3 channels , 128x128
  batch_size=8
  )

"""To make things interesting, we excluded 6th character 'Ray' from train set to test our NN on him."""
test_data = mx.io.ImageRecordIter(
  path_imgrec="./data/diff_data_val.rec", # The target record file.
  data_shape=(3, 128, 128),
  batch_size=8
  )

"""Describing the architecture. It's like building lego - one piece comes on top of another one.
Each variable represents symbolic operation in mxnet. Even layers are considered operations (they really are, e.g.
matrix multiplication is Wx + T for fully connected layer).
So we need to build our operations like this:
--= Input Layer (data) =--
--= Fully connected =--
--= Activation Function =--
<repeat X times for each layer> 
--= Softmax Output =--
Done!
"""

#First 'layer', if it can be put that way.
data = mx.sym.var('data')
input = mx.sym.flatten(data = data)

first = mx.sym.FullyConnected(data = input, num_hidden = 100)
firstact = mx.sym.Activation(data=first, act_type='sigmoid')

#Fully connected. Notice that "input" goes as data here.

#Activation function

#
# fc1 = mx.sym.FullyConnected(data = actfunc, num_hidden = 2000)
# actfunc1 = mx.sym.Activation(data=fc1, act_type='sigmoid')
#
# fc2 = mx.sym.FullyConnected(data = actfunc1, num_hidden = 800)
# actfunc2 = mx.sym.Activation(data=fc2, act_type='sigmoid')

#Going to output
fc = mx.sym.FullyConnected(data = firstact, num_hidden = 42)
softmax = mx.sym.SoftmaxOutput(data = fc, name = 'softmax')
net_model = mx.mod.Module(symbol = softmax, context = mx.gpu())

net_model.fit(train_data, # тренировочные данные
    eval_data = test_data, # валидационные данные
    optimizer = 'sgd', # метод оптимизации, который используется в ходе обучения
    optimizer_params = {'learning_rate': 0.001}, # параметры метода
    eval_metric = 'acc', # метрика для оценки качества обучения (точность)
    batch_end_callback = mx.callback.Speedometer(200, 200), # вывод прогресса
    num_epoch = 1) # количество тренировочных эпох

acc = mx.metric.Accuracy()
net_model.score(test_data, acc)
net_model.save_checkpoint("mega", 1) #saving trained network to file. Can be loaded later
print(acc)