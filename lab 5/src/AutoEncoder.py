import mxnet as mx
import logging
from ImageReader import *
import time

logging.getLogger().setLevel(logging.DEBUG)

batch_size = 1

train_data = mx.io.ImageRecordIter(
    path_imgrec="../../data/all_emotions_train.rec",  # The target record file.
    data_shape=(3, 128, 128),  # 3 channels , 128x128
    batch_size=1
)

test_data = mx.io.ImageRecordIter(
    path_imgrec="../../data/all_emotions_val.rec",  # The target record file.
    data_shape=(3, 128, 128),
    batch_size=1
)

print("Loading pictures...")
X, y = load_pictures('../../data/Emotions/')
X_train, X_test, y_train, y_test = split_data(X, y, 80)
print("Done!")

encoder_iter = mx.io.NDArrayIter(X_train,
                                 y_train,
                                 batch_size, shuffle=True)
encoder_val_iter = mx.io.NDArrayIter(X_test,
                                     y_test,
                                     batch_size)

data = mx.sym.var('data')
#data = mx.sym.flatten(data=data)

encoder = mx.symbol.FullyConnected(data=data, num_hidden=1000, name='encoder')
en_act = mx.sym.Activation(data=encoder, act_type='sigmoid', name='encoder_activation')

decoder = mx.symbol.FullyConnected(data=en_act, num_hidden=128 * 128 * 3, name='decoder')
linreg = mx.sym.LinearRegressionOutput(data=decoder, name='softmax')

lenet_model = mx.mod.Module(symbol=linreg, context=mx.gpu())
#lenet_model.bind(data_shapes=encoder_iter.provide_data, label_shapes=encoder_iter.provide_label)
t = time.clock()
lenet_model.fit(encoder_iter,
                eval_data=encoder_val_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate': 0.1},
                eval_metric='mse',
                batch_end_callback=mx.callback.Speedometer(batch_size, 200),
                num_epoch=1)
print("Clock time difference: %f" % (time.clock() - t))
params = lenet_model.get_params()
prob = lenet_model.predict(encoder_val_iter)
# predict accuracy for lenet
acc = mx.metric.Accuracy()
lenet_model.score(encoder_val_iter, acc)
print(acc)
