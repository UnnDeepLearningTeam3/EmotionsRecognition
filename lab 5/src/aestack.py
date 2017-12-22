import numpy as np

import mxnet as mx
import time

def custom_metric(label, pred):
    label = label.reshape(-1, 128*128*3)
    return np.mean((label - pred) ** 2)
my_eval_metric = mx.metric.create(custom_metric)

def magic(layer_sizes, X_train, initial_size=128*128*3, batch_size=1):
    # init
    y_train = X_train
    params = {}
    encoder_iter = mx.io.NDArrayIter(X_train,
                                     y_train,
                                     batch_size, shuffle=True)
    data = mx.sym.var('data')
    data = mx.sym.flatten(data=data)

    names = ['layer_' + str(x) for x in layer_sizes]

    # pretrain
    for i, (name, count) in enumerate(zip(names, layer_sizes)):
        layer = mx.symbol.FullyConnected(data=data, num_hidden=count, name=name)
        activation = mx.sym.Activation(data=layer, act_type='sigmoid', name=name+'_activation')

        num_hidden = initial_size if i == 0 else layer_sizes[i-1]

        decoder = mx.symbol.FullyConnected(data=activation, num_hidden=num_hidden, name='decoder')
        linreg = mx.sym.LinearRegressionOutput(data=decoder, name='softmax')
        lenet_model = mx.mod.Module(symbol=linreg, context=mx.gpu())

        t = time.clock()
        print("Pretraining #%d start" % i)
        lenet_model.fit(encoder_iter,
                        optimizer='sgd',
                        optimizer_params={'learning_rate': 0.1},
                        eval_metric= (my_eval_metric if i==0 else 'mse'),
                        batch_end_callback=mx.callback.Speedometer(batch_size, 200),
                        num_epoch=1)
        print("Clock time difference: %f" % (time.clock() - t))
        params.update(lenet_model.get_params()[0].items())
        output = lenet_model.symbol.get_internals()[name+'_activation_output']
        params.update({'data': X_train}.items())
        X_train = output.eval(ctx=mx.cpu(), **params)[0]
        y_train = X_train
        encoder_iter = mx.io.NDArrayIter(X_train,
                                         y_train,
                                         batch_size, shuffle=True)
    return params


def build_network(layers):
    data = mx.sym.var('data')
    data = mx.sym.flatten(data=data)

    layer_activation = data

    for i, num_hidden in enumerate(layers):
        layer_name = 'layer_'+str(num_hidden)
        activation_name = 'layer_'+str(num_hidden) + '_activation'

        layer = mx.symbol.FullyConnected(layer_activation, num_hidden=num_hidden, name=layer_name)
        if i == len(layers)-1:
            layer_activation = mx.sym.SoftmaxOutput(data=layer, name="softmax")
        else:
            layer_activation = mx.sym.Activation(data=layer, act_type='sigmoid', name=activation_name)
    return layer_activation
