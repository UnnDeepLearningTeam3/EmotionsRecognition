import mxnet as mx
import logging
from ImageReader import *
import time
from aestack import magic

logging.getLogger().setLevel(logging.DEBUG)

batch_size = 1

print("Loading pictures...")
X, y = load_pictures('../../data/Emotions/')
X_train, X_test, y_train, y_test = split_data(X, y, 80)
print("Done!")

layers = [2000, 1000, 7]
params = magic(layers, X_train)
del(params['data'])
encoder_iter = mx.io.NDArrayIter(X_train,
                                 y_train,
                                 batch_size, shuffle=True)
encoder_val_iter = mx.io.NDArrayIter(X_test,
                                     y_test,
                                     batch_size)

print ('Building architecture...')

data = mx.sym.var('data')
data = mx.sym.flatten(data=data)

layer1 = mx.symbol.FullyConnected(data=data, num_hidden=1200, name='layer_'+str(layers[0]))
layer1_activation = mx.sym.Activation(data=layer1, act_type='sigmoid', name='layer_'+str(layers[0])+'_activation')

layer2 = mx.symbol.FullyConnected(data=layer1_activation, num_hidden=500, name='layer_'+str(layers[1]))
layer2_activation = mx.sym.Activation(data=layer2, act_type='sigmoid', name='layer_'+str(layers[1])+'_activation')

layer3 = mx.symbol.FullyConnected(data=layer2_activation, num_hidden=7, name='layer_'+str(layers[2]))
softmax = mx.sym.SoftmaxOutput(data=layer3, name="softmax")

model = mx.mod.Module(symbol=softmax, context=mx.gpu())
t = time.clock()
print ("Training start")
model.fit(encoder_iter,
          eval_data=encoder_val_iter,
          arg_params=params,
          optimizer='sgd',
          optimizer_params={'learning_rate': 0.1},
          eval_metric='acc',
          batch_end_callback=mx.callback.Speedometer(batch_size, 200),
          num_epoch=1)
print("Clock time difference: %f" % (time.clock() - t))
acc = mx.metric.Accuracy()
model.score(encoder_val_iter, acc)
print(acc)
