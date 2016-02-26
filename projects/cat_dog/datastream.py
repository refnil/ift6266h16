from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.datasets import DogsVsCats
from transformer import ResizeTransformer


def GetStream(image_size=(32,32), batch_size=32, scheme=ShuffledScheme):
    train = DogsVsCats(('train',), subset=slice(0, 20000))
    test = DogsVsCats(('test',))
    validation = DogsVsCats(('train',), subset=slice(20000, 25000))

    def ResizeAndStream(dataset):
        stream = DataStream.default_stream(dataset, iteration_scheme=scheme(dataset.num_examples, batch_size))
        return ResizeTransformer(stream, image_size)

    rs = ResizeAndStream
    return rs(train), rs(test), rs(validation)


