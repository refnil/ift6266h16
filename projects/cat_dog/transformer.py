from fuel.transformers import SourcewiseTransformer, ExpectsAxisLabels
from PIL import Image
import numpy

class ResizeTransformer(SourcewiseTransformer,ExpectsAxisLabels):

    def __init__(self, data_stream, size = (64,64), **kwargs):
        self.size = size
        kwargs.setdefault('which_sources', "image_features")
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(ResizeTransformer, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return [self._resize(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return self._resize(example, source_name)

    def _resize(self, example, source_name):
        target_width, target_height = self.size
        original_height, original_width = example.shape[-2:]

        if example.ndim != 3:
            raise ValueError("The image dimension is {}".format(example.ndim))
        if original_height != target_height or target_width != original_width:
            example = example.astype('float32')
            example = numpy.array([self._resize_layer(layer) for layer in example])

        return example

    def _resize_layer(self, layer):
        im = Image.fromarray(layer)
        im = im.resize(self.size)
        return numpy.array(im)

import random
def zoom():
    _zoom = random.uniform(0.8,0.9)
    _x = random.random()
    _y = random.random()

    def exec(im):
        size = im.size
        csize = (int(size[0]*_zoom),int(size[1]*_zoom))
        x_mov = int((size[0]-csize[0])*_x)
        y_mov = int((size[1]-csize[1])*_y)
        crop = (x_mov,y_mov,x_mov+csize[0],y_mov+csize[1])
        return im.crop(crop).resize(size)

    return exec

def rotate():
    _rotate = random.randint(-30,30)
    def exec(im):
        return im.rotate(_rotate)
    return exec

def flip():
    return lambda im: im.transpose(Image.FLIP_LEFT_RIGHT)



class DataAugmentation(SourcewiseTransformer,ExpectsAxisLabels):

    def __init__(self, data_stream,  **kwargs):
        kwargs.setdefault('which_sources', "image_features")
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(DataAugmentation, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return [self._update(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return self._update(example, source_name)

    def _update(self, example, source_name):

        if example.ndim != 3:
            raise ValueError("The image dimension is {}".format(example.ndim))

        if random.randint(0,1):
            transformation = random.choice([zoom(),rotate(),flip()])

            example = example.astype('float32')
            example = numpy.array([self._update_layer(layer,transformation) for layer in example])

        return example

    def _update_layer(self, layer, transformation):
        im = Image.fromarray(layer)
        return numpy.array(transformation(im))
