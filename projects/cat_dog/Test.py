# In[3]:

from blocks.bricks import Linear, Softmax, Rectifier
from theano import tensor
from fuel.datasets import DogsVsCats
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream


# In[4]:

from transformer import ResizeTransformer


# In[5]:

dataset = DogsVsCats(('train',), subset=slice(0, 20000))


# In[6]:

astream = DataStream.default_stream(
    dataset,
    iteration_scheme=SequentialScheme(dataset.num_examples, 32)
)


# In[7]:

stream = ResizeTransformer(astream, size=(28,28))


# In[8]:

stream.axis_labels


# In[9]:

data = next(stream.get_epoch_iterator())
im = data[0][0]
print(len(im))
print(len(im[0]))
print(len(im[0][0]))
data[0][0][0][0][0]


# In[10]:

len(data[0][0][1])


# In[11]:

len(data[0][0][0][2])


# In[12]:

from blocks.bricks.conv import Convolutional, MaxPooling


# In[13]:

FirstConvo = Convolutional(name='First Convo', filter_size=(7,7), num_filters=20, num_channels=3)
FirstPooling = MaxPooling(name='First Pooling', pooling_size=(3,3))


# In[14]:

x = tensor.matrix('features')

#from blocks.bricks import Linear, Rectifier, Softmax
input_to_hidden = Linear(name='input_to_hidden', input_dim=2352, output_dim=300)
h = Rectifier().apply(input_to_hidden.apply(x))
hidden_to_output = Linear(name='hidden_to_output', input_dim=300, output_dim=2)
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

from fuel.transformers import Flatten
data_stream = Flatten(stream)

from blocks.algorithms import GradientDescent, Scale
algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Scale(learning_rate=0.1))

mnist_test = DogsVsCats(('test',))
data_stream_test = Flatten(
    ResizeTransformer(
    DataStream.default_stream(
        mnist_test,
        iteration_scheme=SequentialScheme(mnist_test.num_examples, batch_size=1024))
    ,(28,28)))

from blocks.extensions.monitoring import DataStreamMonitoring
monitor = DataStreamMonitoring(variables=[cost], data_stream=data_stream_test, prefix="test")

from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                     extensions=[monitor, FinishAfter(after_n_epochs=1), Printing()])


# In[ ]:

main_loop.run()

