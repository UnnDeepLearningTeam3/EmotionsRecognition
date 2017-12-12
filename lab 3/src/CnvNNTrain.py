import mxnet as mx
import logging

import time

logging.getLogger().setLevel(logging.DEBUG)

batch_size = 4

train_data = mx.io.ImageRecordIter(
  path_imgrec="../../data/all_emotions_train.rec", # The target record file.
  data_shape=(3, 128, 128), #3 channels , 128x128
  batch_size=4
  )

test_data = mx.io.ImageRecordIter(
  path_imgrec="../../data/all_emotions_val.rec", # The target record file.
  data_shape=(3, 128, 128),
  batch_size=4
  )

data = mx.sym.var('data')

#dataf = mx.sym.flatten(data=data)

# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(7,7), num_filter=500, stride=(4,4))
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(3,3), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=500, stride=(1,1))
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(3,3), stride=(2,2))
# third layer
conv3 = mx.sym.Convolution(data=pool2, kernel=(3,3), num_filter=500, stride=(1,1))
tanh3 = mx.sym.Activation(data=conv3, act_type="tanh")
pool3 = mx.sym.Pooling(data=tanh3, pool_type="max", kernel=(3,3), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=pool3)

fc0 = mx.symbol.FullyConnected(data=flatten, num_hidden=1024)
tanh4 = mx.sym.Activation(data=fc0, act_type="tanh")

fc1 = mx.symbol.FullyConnected(data=tanh4, num_hidden=512)
tanh5 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh5, num_hidden=7)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

# create a trainable module on GPU 0
lenet_model = mx.mod.Module(symbol=lenet, context=mx.gpu())
# train with the same

t = time.clock()
lenet_model.fit(train_data,
                eval_data=test_data,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.01},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 200),
                num_epoch=1)
print("Clock time difference: %f" % (time.clock()-t))
prob = lenet_model.predict(test_data)
# predict accuracy for lenet
acc = mx.metric.Accuracy()
lenet_model.score(test_data, acc)
print(acc)
