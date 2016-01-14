import cPickle
import gzip

train_set, valid_set, test_set = None, None, None
with gzip.open('../../dataset/mnist.data', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)
