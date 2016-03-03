import argparse
from fuel.streams import ServerDataStream
from datastream import get_dvc, get_mnist
import os
import time

from theano import tensor
from blocks.bricks import Rectifier, Softmax
from blocks.bricks import MLP
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter

from bricks import FinishIfNoImprovementAfterPlus, CheckpointBest


def valid(net, train_stream, test_stream, **kwargs):
    pass

def train_net(net, train_stream, test_stream, L2=False, early_stopping=False,
        finish=None,
        **ignored):
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')

    y_hat = net.apply(x)

    #Cost
    cost_before = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
    cost_before.name = "cost_without_regularization"

    #Regularization
    cg = ComputationGraph(cost_before)
    WS = VariableFilter(roles=[WEIGHT])(cg.variables)

    if L2:
        L2 = sum([0.005 * (W ** 2).sum() for W in WS])
        L2.name = "L2 regularization"
        cost_before += L2

    cost = cost_before
    cost.name = 'cost_with_regularization'

    #Initialization
    net.initialize()

    #Algorithm
    algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Scale(learning_rate=0.1))

    extensions = []

    #Monitoring
    monitor = DataStreamMonitoring(variables=[cost], data_stream=test_stream, prefix="test")
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
        Printing(every_n_epochs=5, after_epoch=None)
        ])

    #Main loop
    main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm, extensions=extensions)

    main_loop.run()

def check():
    pass

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-p', '--parallel', action='store_true')
    parser.add_argument('-m', '--mnist', action='store_true')
    parser.add_argument('--L2', action='store_true')
    parser.add_argument('-e', '--early_stopping', action='store_true')
    parser.add_argument('--finish', type=int)
    args = parser.parse_args()

    if args.mnist:
        train, test = get_mnist()
        net = MLP(activations=[Rectifier(), Softmax()],
                  dims=[784, 100, 10],
                  weights_init=IsotropicGaussian(),
                  biases_init=Constant(0.01))
    else:
        if args.parallel:
            train = ServerDataStream(('train',), True, port=5571)
            test = ServerDataStream(('test',), True, port=5572)
            validation = ServerDataStream(('train'), True, port=5573)
        else:
            train, test, validation = get_dvc()

    train_net(net, train, test, **vars(args))
