import mxnet as mx
import logging
from ImageReader import *
import time
from aestack import magic, build_network

logging.getLogger().setLevel(logging.DEBUG)

batch_size = 1

print("Loading pictures...")
X, y = load_pictures('../../data/Emotions/')
X_train, X_test, y_train, y_test = split_data(X, y, 80)
print("Done!")

layers = [1200, 700, 200, 7]
params = magic(layers, X_train)
del(params['data'])
encoder_iter = mx.io.NDArrayIter(X_train,
                                 y_train,
                                 batch_size, shuffle=True)
encoder_val_iter = mx.io.NDArrayIter(X_test,
                                     y_test,
                                     batch_size)

print ('Building architecture...')

arch = build_network(layers)

model = mx.mod.Module(symbol=arch, context=mx.gpu())
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
