from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.datasets import DogsVsCats, MNIST
from fuel.transformers import Flatten
from transformer import ResizeTransformer, DataAugmentation

def get_dvc(image_size=(32,32), trainning=True, shortcut=False,augmentation=False):

    if shortcut:
        subset_train = slice(0,35)
        subset_validation = slice(22500,22505)
    else:
        subset_train = slice(0,22500)
        subset_validation = slice(22500,25000)


    train = DogsVsCats(('train',), subset=subset_train)
    test = DogsVsCats(('test',), )
    validation = DogsVsCats(('train',), subset=subset_validation)

    def create_dataset(dataset):
        if trainning:
            scheme = ShuffledScheme(dataset.num_examples, 64)
        else:
            scheme = SequentialScheme(dataset.num_examples, 64)
        stream = DataStream.default_stream(dataset, iteration_scheme=scheme)
        return ResizeTransformer(stream, image_size)

    streams = list(map(create_dataset, [train,test,validation]))
    if augmentation:
        streams[0] = DataAugmentation(streams[0])
    return streams


def get_mnist():
    mnist = MNIST(("train",))
    mnist_test = MNIST(("test",))

    def s(s):
        return Flatten(DataStream.default_stream(
            s,
            iteration_scheme=ShuffledScheme(s.num_examples, batch_size=256)))

    return s(mnist), s(mnist_test)

