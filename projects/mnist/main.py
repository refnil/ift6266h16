import mnist
import numpy as np
import imaging

class Layer:
    def __init__(self):
        pass

    def forward(self, input_):
        pass

    def backward(self, gradient):
        pass

    def update(self, gradient, learning):
        pass

class BiasLayer(Layer)
    def __init__(self, nb_node):
        self.b = nb.zeroes(nb_node)

    def forward(self, input_):
        return input_ = self.b

    def backward(self, output_):
        pass

class WeightLayer(Layer):
    def __init__(self, nb_input, nb_node):
        self.w = np.random.randn(nb_input,nb_node)

    def forward(self, input_):
        return np.dot(input_,self.w)

    def backward(self, output_):
        return gradient_wrt_output.dot(W.T)

class Sigmoid(Layer):
    def forward(self, input_):
        return 1/(1+nb.exp(-input_))

class Softmax(Layer):
    def forward(self, input_):


class MultiLayer(Layer):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input_):
        pass

    def backward(self, output_):
        pass

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def run(self, data):
        return self._run(data)[-1]

    def _run(self, data):
        inputs = [data]
        for layers in self.layers:
            inputs.append(layers.forward(inputs[-1]))

        return inputs

def split(set_, size):
    while len(set_) > 0:
        yield set_[:size]
        set_ = set_[size:]

def main():
    layers = [LinearLayer(784, 10)]
    nn = NeuralNetwork(layers)
    data = list(split(mnist.train_set[0], 6))[0]
    imaging.combine_mnist(data).save("test.bmp")
    o = nn.run(data)
    print(o)

if __name__=="__main__":
    main()
