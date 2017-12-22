import copy

import mxnet as mx
import logging

import time

logging.getLogger().setLevel(logging.DEBUG)

batch_size = 4

mnist = mx.test_utils.get_mnist()
train_mnist = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_mnist = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

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


# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=20, name='second_conv')
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=pool2)

fc0 = mx.symbol.FullyConnected(data=flatten, num_hidden=120)
tanh4 = mx.sym.Activation(data=fc0, act_type="tanh")
# second fullc
fc3 = mx.sym.FullyConnected(data=tanh4, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

# create a trainable module on GPU 0
lenet_model = mx.mod.Module(symbol=lenet, context=mx.gpu())
# train with the same

t = time.clock()
lenet_model.fit(train_mnist,
                eval_data=val_mnist,
                optimizer='sgd',
                optimizer_params={'learning_rate': 0.1},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 200),
                num_epoch=1)
print("Clock time difference: %f" % (time.clock()-t))
acc = mx.metric.Accuracy()
lenet_model.score(val_mnist, acc)
print(acc)


#weights
params = lenet_model.get_params()[0]
bias = copy.deepcopy(params['second_conv_bias'])
weight = copy.deepcopy(params['second_conv_weight'])

fc0 = mx.symbol.FullyConnected(data=flatten, num_hidden=1000)
tanh4 = mx.sym.Activation(data=fc0, act_type="tanh")
fc0 = mx.symbol.FullyConnected(data=tanh4, num_hidden=500)
tanh4 = mx.sym.Activation(data=fc0, act_type="tanh")
fc0 = mx.symbol.FullyConnected(data=tanh4, num_hidden=200)
tanh4 = mx.sym.Activation(data=fc0, act_type="tanh")
fc3 = mx.sym.FullyConnected(data=tanh4, num_hidden=7)
lenet = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
lenet_model = mx.mod.Module(symbol=lenet, context=mx.gpu())

lenet_model.bind(data_shapes=train_data.provide_data,
         label_shapes=train_data.provide_label)
init = mx.init.Normal(1.0)
lenet_model.init_params(init, force_init=True)
params = lenet_model.get_params()[0]
params['second_conv_bias'] = bias
params['second_conv_weight'] = weight
t = time.clock()
lenet_model.fit(train_data,
                eval_data=test_data,
                optimizer='sgd',
                optimizer_params={'learning_rate': 0.1},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 200),
                num_epoch=1,
                arg_params=params)
print("Clock time difference: %f" % (time.clock()-t))
prob = lenet_model.predict(test_data)
# predict accuracy for lenet
acc = mx.metric.Accuracy()
lenet_model.score(test_data, acc)
print(acc)