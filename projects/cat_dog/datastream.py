from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.datasets import DogsVsCats, MNIST
from fuel.transformers import Flatten
from transformer import ResizeTransformer


def get_dvc(image_size=(32,32), batch_size=32, scheme=ShuffledScheme):
    train = DogsVsCats(('train',), subset=slice(0, 20000))
    test = DogsVsCats(('test',))
    validation = DogsVsCats(('train',), subset=slice(20000, 25000))

    def ResizeAndStream(dataset):
        stream = DataStream.default_stream(dataset, iteration_scheme=scheme(dataset.num_examples, batch_size))
        return ResizeTransformer(stream, image_size)

    rs = ResizeAndStream
    return rs(train), rs(test), rs(validation)


def get_mnist():
    mnist = MNIST(("train",))
    mnist_test = MNIST(("test",))

    def s(s):
        return Flatten(DataStream.default_stream(
            s,
            iteration_scheme=ShuffledScheme(s.num_examples, batch_size=256)))

    return s(mnist), s(mnist_test)

