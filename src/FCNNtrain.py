import mxnet as mx
import logging

logging.getLogger().setLevel(logging.DEBUG)

"""Train data consist of 5 characters with 7 different emotions we need to recognize"""
train_data = mx.io.ImageRecordIter(
  path_imgrec="./data/data_info_train.rec", # The target record file.
  data_shape=(3, 128, 128), #3 channels , 128x128
  batch_size=8
  )

"""To make things interesting, we excluded 6th character 'Ray' from train set to test our NN on him."""
test_data = mx.io.ImageRecordIter(
  path_imgrec="./data/ray_validation.rec", # The target record file.
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

#Fully connected. Notice that "input" goes as data here.
fc = mx.sym.FullyConnected(data = input, num_hidden = 100)
#Activation function
actfunc = mx.sym.Activation(data=fc, act_type='sigmoid')

fc1 = mx.sym.FullyConnected(data = actfunc, num_hidden = 100)
actfunc1 = mx.sym.Activation(data=fc1, act_type='sigmoid')

fc2 = mx.sym.FullyConnected(data = actfunc1, num_hidden = 100)
actfunc2 = mx.sym.Activation(data=fc2, act_type='sigmoid')

#Going to output
softmax = mx.sym.SoftmaxOutput(data = actfunc2, name = 'softmax')
net_model = mx.mod.Module(symbol = softmax, context = mx.cpu())

net_model.fit(train_data, # тренировочные данные
    eval_data = test_data, # валидационные данные
    optimizer = 'sgd', # метод оптимизации, который используется в ходе обучения
    optimizer_params = {'learning_rate': 0.1}, # параметры метода
    eval_metric = 'acc', # метрика для оценки качества обучения (точность)
    batch_end_callback = mx.callback.Speedometer(200, 200), # вывод прогресса
    num_epoch = 5) # количество тренировочных эпох

acc = mx.metric.Accuracy()
net_model.score(test_data, acc)
net_model.save_checkpoint("conf0", 1) #saving trained network to file. Can be loaded later
print(acc)