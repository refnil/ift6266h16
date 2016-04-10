import argparse
from fuel.streams import ServerDataStream
from datastream import get_dvc, get_mnist
import os
import time
import numpy as np

from theano import tensor
from blocks.bricks import Rectifier, Softmax
from blocks.bricks import MLP, FeedforwardSequence
from blocks.bricks.conv import ConvolutionalSequence, Convolutional, MaxPooling, Flattener
from blocks.initialization import IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.initialization import Constant, Uniform

from bricks import FinishIfNoImprovementAfterPlus, CheckpointBest

def train_net(net, train_stream, test_stream, L1 = False, L2=False, early_stopping=False,
        finish=None, dropout=False,
        **ignored):
    x = tensor.tensor4('image_features')
    y = tensor.lmatrix('targets')

    y_hat = net.apply(x)

    #Cost
    cost_before = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
    cost_before.name = "cost_without_regularization"

    #Error
    #Taken from brodesf
    error = MisclassificationRate().apply(y.flatten(), y_hat)
    error.name = "Misclassification rate"

    #Regularization
    cg = ComputationGraph(cost_before)
    WS = VariableFilter(roles=[WEIGHT])(cg.variables)

    if dropout:
        cg = apply_dropout(cg, WS, 0.5)

    if L1:
        L1_reg = 0.005 * sum([abs(W).sum() for W in WS])
        L1_reg.name = "L1 regularization"
        cost_before += L1_reg

    if L2:
        L2_reg = 0.005 * sum([(W ** 2).sum() for W in WS])
        L2_reg.name = "L2 regularization"
        cost_before += L2_reg

    cost = cost_before
    cost.name = 'cost_with_regularization'

    #Initialization
    net.initialize()

    #Algorithm
    algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Scale(learning_rate=0.1))

    extensions = []

    #Monitoring
    monitor = DataStreamMonitoring(variables=[cost, error], data_stream=test_stream, prefix="test")
    extensions.append(monitor)

    def filename(suffix=""):
        return "checkpoints/" + str(os.getpid()) + "_" + str(time.time()) + suffix + ".tar"

    #Serialization
    serialization = Checkpoint(filename())
    extensions.append(serialization)

    notification = "test_cost_with_regularization"
    track = TrackTheBest(notification)
    checkpointbest = CheckpointBest(notification, filename("_best"))
    extensions.extend([track, checkpointbest])

    if early_stopping:
        stopper = FinishIfNoImprovementAfterPlus("test_cost_with_regularization_best_so_far")
        extensions.append(stopper)

    #Other extensions
    if finish != None:
        extensions.append(FinishAfter(after_n_epochs=finish))

    extensions.extend([
        Timing(),
        Printing()
        ])

    #Main loop
    main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm, extensions=extensions)

    main_loop.run()

def net_dvc(image_size=(32,32)):
    convos = [5,5]
    pools = [3,3]
    filters = [35,50]

    tuplify = lambda x: (x,x)
    convos = list(map(tuplify, convos))
    conv_layers = [Convolutional(filter_size=s,num_filters=o, num_channels=i) for s,o,i in zip(convos, filters, [3] + filters)]

    pool_layers = [MaxPooling(p) for p in map(tuplify, pools)]

    activations = [Rectifier() for i in convos]

    layers = [i for l in zip(conv_layers, activations, pool_layers) for i in l]

    cnn = ConvolutionalSequence(layers, 3,  image_size=image_size, name="cnn",
            weights_init=Uniform(width=.2),
            biases_init=Constant(0))

    cnn._push_allocation_config()
    cnn_output = np.prod(cnn.get_dim('output'))

    mlp_size = [cnn_output,20,2]
    mlp = MLP([Rectifier(), Softmax()], mlp_size,  name="mlp",
            weights_init=Uniform(width=.2),
            biases_init=Constant(0))

    seq = FeedforwardSequence([net.apply for net in [cnn,Flattener(),mlp]])
    seq.push_initialization_config()

    seq.initialize()
    return seq


def net_mnist():
    return  MLP(activations=[Rectifier(), Softmax()],
              dims=[784, 100, 10],
              weights_init=IsotropicGaussian(),
              biases_init=Constant(0.))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-p', '--parallel', action='store_true')
    parser.add_argument('-m', '--mnist', action='store_true')
    parser.add_argument('--L1', action='store_true')
    parser.add_argument('--L2', action='store_true')
    parser.add_argument('-e', '--early_stopping', action='store_true')
    parser.add_argument('-d', '--dropout', action='store_true')
    parser.add_argument('--finish', type=int)
    parser.add_argument('--port', default= 5557, type=int)
    args = parser.parse_args()

    if args.mnist:
        train, test = get_mnist()
        net = net_mnist()
    else:
        net = net_dvc()
        if args.parallel:
            sources = ('image_features','targets')
            train = ServerDataStream(sources, True, port=args.port)
            valid = ServerDataStream(sources, True, port=args.port+1)
            test = ServerDataStream(sources, True, port=args.port+2)
        else:
            train, valid, test = get_dvc()

    train_net(net, train, test, **vars(args))
