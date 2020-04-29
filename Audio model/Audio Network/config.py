#coding:utf8
import warnings
from torchvision import transforms

class DefaultConfig(object):
    dataset_name = 'imagenet'
    data_path = '/home/liuhan/C3D/Mel_spectrum_resize/'
    in_channels = 3
    img_rows = 224
    num_classes = 7
    model_name = 'ResNet'
    model_save_path = '/home/liuhan/C3D/pytorch-classification-master/checkpoints/ResNet.ckpt'
    
    batch_size = 8
    test_batch_size = 5
    lr = 1e-3
    momentum = 0.5
    max_epochs = 100
    save_freq = 50

    def parse(self, kwargs):
        for k, v in kwargs.iteritems():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute {}".format(k))
                setattr(self, k, v)
        print('user config:')
        for k, v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                print (k, getattr(self, k))

opt = DefaultConfig()
