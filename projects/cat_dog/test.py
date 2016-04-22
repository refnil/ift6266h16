
# In[15]:

from blocks.bricks import Linear, Softmax, Rectifier
from theano import tensor
from fuel.datasets import DogsVsCats
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream


# In[16]:

from transformer import ResizeTransformer


# In[46]:

dataset = DogsVsCats(('train',), subset=slice(0, 20000))


# In[18]:

astream = DataStream.default_stream(
    dataset,
    iteration_scheme=SequentialScheme(dataset.num_examples, 32)
)


# In[19]:

stream = ResizeTransformer(astream, size=(28,28))


# In[20]:

stream.axis_labels


# In[43]:

data = next(stream.get_epoch_iterator())
im = data[0][0]
print(len(data[0]))
print(len(im))
print(len(im[0]))
print(len(im[0][0]))
data[0][0][0][0][0]


# In[24]:

from blocks.bricks.conv import Convolutional, MaxPooling


# In[25]:

FirstConvo = Convolutional(name='First Convo', filter_size=(7,7), num_filters=20, num_channels=3)
FirstPooling = MaxPooling(name='First Pooling', pooling_size=(3,3))


# In[26]:

from fuel.datasets import MNIST
mnist = MNIST(("train",))

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten
data_stream_mnist = Flatten(DataStream.default_stream(
    mnist,
    iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=256)))

mnist_test = MNIST(("test",))
data_stream_test_mnist = Flatten(DataStream.default_stream(
     mnist_test,
     iteration_scheme=SequentialScheme(
         mnist_test.num_examples, batch_size=32000)))


# In[27]:

mnist = True
if mnist:
    ds = data_stream_mnist
    dst = data_stream_test_mnist
else:
    ds = Flatten(stream)
    mnist_test = DogsVsCats(('test',))
    dst = Flatten(ResizeTransformer(
        DataStream.default_stream(
            mnist_test,
            iteration_scheme=SequentialScheme(mnist_test.num_examples, batch_size=1024))
                ,(28,28)))


# In[40]:

x = tensor.matrix('features')

from blocks.bricks import Linear, Rectifier, Softmax
input_to_hidden = Linear(name='input_to_hidden', input_dim=784, output_dim=50)
h = Rectifier().apply(input_to_hidden.apply(x))
hidden_to_output = Linear(name='hidden_to_output', input_dim=50, output_dim=10)
y_hat = Softmax().apply(hidden_to_output.apply(h))

y = tensor.lmatrix('targets')
from blocks.bricks.cost import CategoricalCrossEntropy
cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)

from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
cg = ComputationGraph(cost)
W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)
cost = cost + 0.005 * (W1 ** 2).sum() + 0.005 * (W2 ** 2).sum()
cost.name = 'cost_with_regularization'

from blocks.initialization import IsotropicGaussian, Constant
input_to_hidden.weights_init = hidden_to_output.weights_init = IsotropicGaussian(0.01)
input_to_hidden.biases_init = hidden_to_output.biases_init = Constant(0)
input_to_hidden.initialize()
hidden_to_output.initialize()

from blocks.algorithms import GradientDescent, Scale
algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Scale(learning_rate=0.1))

from blocks.extensions.monitoring import DataStreamMonitoring
monitor = DataStreamMonitoring(variables=[cost], data_stream=dst, prefix="test")

from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
main_loop = MainLoop(data_stream=ds, algorithm=algorithm,
                     extensions=[monitor, FinishAfter(after_n_epochs=10), Printing(every_n_epochs=5, after_epoch=None)])


# In[41]:

main_loop.run()



