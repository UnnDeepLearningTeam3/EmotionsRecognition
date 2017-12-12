import mxnet as mx
import logging

import time

logging.getLogger().setLevel(logging.DEBUG)

train_data = mx.io.ImageRecordIter(
  path_imgrec="../../data/all_emotions_train.rec", # The target record file.
  data_shape=(3, 128, 128), #3 channels , 128x128
  batch_size=8
  )

test_data = mx.io.ImageRecordIter(
  path_imgrec="../../data/all_emotions_val.rec", # The target record file.
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

first = mx.sym.FullyConnected(data = input, num_hidden = 2500)
firstact = mx.sym.Activation(data=first, act_type='sigmoid')

second = mx.sym.FullyConnected(data = firstact, num_hidden = 1000)
secondact = mx.sym.Activation(data=second, act_type='sigmoid')

third = mx.sym.FullyConnected(data = secondact, num_hidden = 500)
thirdact = mx.sym.Activation(data=third, act_type='sigmoid')

#Fully connected. Notice that "input" goes as data here.

#Activation function

#
# fc1 = mx.sym.FullyConnected(data = actfunc, num_hidden = 2000)
# actfunc1 = mx.sym.Activation(data=fc1, act_type='sigmoid')
#
# fc2 = mx.sym.FullyConnected(data = actfunc1, num_hidden = 800)
# actfunc2 = mx.sym.Activation(data=fc2, act_type='sigmoid')

#Going to output
fc = mx.sym.FullyConnected(data = thirdact, num_hidden = 7)
softmax = mx.sym.SoftmaxOutput(data = fc, name = 'softmax')
net_model = mx.mod.Module(symbol = softmax, context = mx.gpu())

t = time.clock()

net_model.fit(train_data,
    eval_data = test_data,
    optimizer = 'sgd',
    optimizer_params = {'learning_rate': 0.001},
    eval_metric = 'acc',
    batch_end_callback = mx.callback.Speedometer(200, 200),
    num_epoch = 1)

print("Clock time difference: %f" % (time.clock()-t))
acc = mx.metric.Accuracy()
net_model.score(test_data, acc)
#net_model.save_checkpoint("mega", 1) #saving trained network to file. Can be loaded later
print(acc)