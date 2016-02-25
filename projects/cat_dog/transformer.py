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
        if original_height != target_height and target_width != original_width:
            example = example.astype('float32')
            example = numpy.array([self._resize_layer(layer) for layer in example])
        return example

    def _resize_layer(self, layer):
        im = Image.fromarray(layer)
        im = im.resize(self.size)
        return numpy.array(im)
