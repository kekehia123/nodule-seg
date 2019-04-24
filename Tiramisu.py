import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['FCDenseNet', 'FCDensenet56', 'FCDensenet67', 'FCDensenet103']


class FCDenseNet(nn.Module):
    r"""
    The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
    https://arxiv.org/abs/1611.09326

    In this paper, we extend DenseNets to deal with the problem of semantic segmentation. We achieve state-of-the-art
    results on urban scene benchmark datasets such as CamVid and Gatech, without any further post-processing module nor
    pretraining. Moreover, due to smart construction of the model, our approach has much less parameters than currently
    published best entries for these datasets.
    """

    def __init__(self,
                 num_input_features: int = 1,                
                 num_classes: int = 1000,
                 growth_rate: int = 16,
                 drop_rate: float = 0.2,

                 num_pool = 5,
                 num_layers_per_blocks = (4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4),
                 
                 num_init_features: int = 48,
                 dense_bn_size = None,
                 
                 dense_compression = 1.0,
                 ):
        
        super(FCDenseNet, self).__init__()

        assert 0 < dense_compression <= 1, 'compression of densenet should be between 0 and 1'
        self.num_classes = num_classes

        if isinstance(num_layers_per_blocks, (tuple, list)):
            assert (len(num_layers_per_blocks) == 2 * num_pool + 1)
        elif type(num_layers_per_blocks) == int:
            num_layers_per_blocks = [num_layers_per_blocks] * (2 * num_pool + 1)
        else:
            raise ValueError
        
        # ==== first convolution ==== 
        
        self.add_module('conv0', nn.Conv2d(num_input_features, num_init_features, kernel_size=3, padding=1, bias=False))
        
        current_channels = num_init_features
        skip_connections_channels = []

        # ==== Downsampling path ====        
        
        self.down_dense = nn.Module()
        self.down_trans = nn.Module()
        
        for i in range(num_pool):
            # Dense Block
            block = _DenseBlock(num_layers=num_layers_per_blocks[i], num_input_features=current_channels,  
                                growth_rate=growth_rate, concat_input=True, bn_size=dense_bn_size, drop_rate=drop_rate)
            current_channels = block.out_channels
            self.down_dense.add_module('denseblock_%d' % i, block)

            skip_connections_channels.append(block.out_channels)

            transition = _Transition_Down(num_input_features=current_channels, 
                                          num_output_features=int(current_channels * dense_compression), drop_rate=drop_rate)
            current_channels = transition.out_channels
            self.down_trans.add_module('trans_%d' % i, transition)
        
        # ===== Middle Bottleneck ====
        # Renamed from "bottleneck" in the paper, to avoid confusion with the Bottleneck of DenseLayers

        self.middle = _DenseBlock(num_layers=num_layers_per_blocks[num_pool], num_input_features=current_channels,
                                  growth_rate=growth_rate, concat_input=False, bn_size=dense_bn_size, drop_rate=drop_rate)
            # Note: a little different from https://github.com/baldassarreFe/pytorch-densenet-tiramisu/blob/master/dense/fc_densenet/fc_densenet.py
            # I think in the middle layer, concat_input should be set to False

        current_channels = self.middle.out_channels

        # ==== Upsampling path ====

        skip_connections_channels = skip_connections_channels[: : -1]

        self.up_dense = nn.Module()
        self.up_trans = nn.Module()
        for i in range(num_pool):
            transition = _Transition_Up(upsample_channels=current_channels, skip_channels = skip_connections_channels[i])
            current_channels = transition.out_channels
            self.up_trans.add_module('trans_%d' % i, transition)
            
            concat_input = True if i == num_pool-1 else False

            block = _DenseBlock(num_layers=num_layers_per_blocks[num_pool+1 + i], num_input_features=current_channels,
                                growth_rate=growth_rate, concat_input=concat_input, bn_size=dense_bn_size, drop_rate=drop_rate)
            current_channels = block.out_channels
            self.up_dense.add_module('denseblock_%d' % i, block)
        
        # ==== Softmax ====

        self.finalconv = nn.Conv2d(current_channels, num_classes, kernel_size=1, stride=1, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.conv0(x)
        #print('input:', x.size())

        skip_tensors = []
        for dense, trans in zip(self.down_dense.children(), self.down_trans.children()):
            out = dense(out)
            skip_tensors.append(out)
            out = trans(out)
        
        out = self.middle(out)

        for trans, dense in zip(self.up_trans.children(), self.up_dense.children()):
            skip = skip_tensors.pop()
            out = trans(out, skip)
            out = dense(out)
            
        #x_out = self.softmax(out)
        
        #print('before softmax:', out.size())
        
        if self.num_classes > 1:
            x_out = F.log_softmax(self.finalconv(out), dim=1)
        else:
            x_out = self.finalconv(out)
        
            
        assert x_out.size()[2:] == x.size()[2:]
        #print('output:', x_out.size())
        return x_out
    
    def predict(self, x):
        logits = self(x)
        return F.softmax(logits)
    
def FCDensenet46(num_classes, dense_bn_size=None, dense_compression=1.0):
    model = FCDenseNet(num_classes=num_classes, growth_rate=8, num_pool=4, num_layers_per_blocks=4, 
                       dense_bn_size=dense_bn_size, dense_compression=dense_compression)
    return model

def FCDensenet45(num_classes, dense_bn_size=None, dense_compression=1.0):
    model = FCDenseNet(num_classes=num_classes, growth_rate=12, num_layers_per_blocks=3, 
                       dense_bn_size=dense_bn_size, dense_compression=dense_compression)
    return model

def FCDensenet56(num_classes, dense_bn_size=None, dense_compression=1.0):
    model = FCDenseNet(num_classes=num_classes, growth_rate=12, num_layers_per_blocks=4, 
                       dense_bn_size=dense_bn_size, dense_compression=dense_compression)
    return model

def FCDensenet67(num_classes, dense_bn_size=None, dense_compression=1.0):
    model = FCDenseNet(num_classes=num_classes, growth_rate=16, num_layers_per_blocks=5, 
                       dense_bn_size=dense_bn_size, dense_compression=dense_compression)
    return model

def FCDensenet103(num_classes, dense_bn_size=None, dense_compression=1.0):
    model = FCDenseNet(num_classes=num_classes, growth_rate=16, num_layers_per_blocks=(4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4), 
                       dense_bn_size=dense_bn_size, dense_compression=dense_compression)
    return model

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features: int, growth_rate: int,
                 bn_size = None, drop_rate = 0.2):
        super(_DenseLayer, self).__init__()
#         self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('norm', nn.InstanceNorm2d(num_input_features, affine=True))
        self.add_module('relu', nn.ReLU(inplace=True))
        in_channels = num_input_features

        if bn_size and isinstance(bn_size, int):
            self.add_module('conv_bottleneck', nn.Conv2d(num_input_features, bn_size * growth_rate, 
                                                         kernel_size=1, stride=1, bias=False))
#             self.add_module('norm_bottleneck', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('norm_bottleneck', nn.InstanceNorm2d(bn_size * growth_rate, affine=True))
            self.add_module('relu_bottleneck', nn.ReLU(inplace=True))
            in_channels = bn_size * growth_rate

        self.add_module('conv', nn.Conv2d(in_channels, growth_rate,
                                                kernel_size=3, stride=1, padding=1, bias=False))
        
        if drop_rate > 0:
            self.add_module('drop', nn.Dropout2d(drop_rate, inplace=True))
    
    def forward(self, x):
        return super().forward(x)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, 
                 concat_input: bool = False, bn_size = None, drop_rate: float = 0.2):
        super(_DenseBlock, self).__init__()

        self.concat_input = concat_input
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        self.num_input_features = num_input_features
        self.out_channels = growth_rate * num_layers
        if self.concat_input:
            self.out_channels += self.num_input_features

        for i in range(num_layers):
            layer = _DenseLayer(num_input_features=num_input_features + i * growth_rate, 
                                growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
    
    def forward(self, x):
        layer_input = x
        
        all_outputs = [x] if self.concat_input else []        
        for layer in self._modules.values():
            new_features = layer(layer_input)
            layer_input = torch.cat([layer_input, new_features], dim=1)
            all_outputs.append(new_features)
        
        return torch.cat(all_outputs, dim=1)
    
class _Transition_Down(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate: float = 0.2):
        super(_Transition_Down, self).__init__()
        self.out_channels = num_output_features
        
#         self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('norm', nn.InstanceNorm2d(num_input_features, affine=True))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        
        if drop_rate > 0:
            self.add_module('drop', nn.Dropout2d(drop_rate))

        self.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))
        
class _Transition_Up(nn.Module):
    r"""
    Transition Up Block as described in [FCDenseNet](https://arxiv.org/abs/1611.09326)

    The block upsamples the feature map and concatenates it with the feature map coming from the skip connection.
    If the two maps don't overlap perfectly they are first aligened centrally and cropped to match.
    """

    def __init__(self, upsample_channels: int, skip_channels = None):
        r"""
        :param upsample_channels: number of channels from the upsampling path
        :param skip_channels: number of channels from the skip connection, it is not required,
                              but if specified allows to statically compute the number of output channels
        """
        super(_Transition_Up, self).__init__()

        self.upsample_channels = upsample_channels
        self.skip_channels = skip_channels
        self.out_channels = upsample_channels + skip_channels if skip_channels is not None else None

        self.add_module('upconv', nn.ConvTranspose2d(self.upsample_channels, self.upsample_channels,
                                                     kernel_size=3, stride=2, padding=0, bias=True))
        self.add_module('concat', CenterCropConcat())
    
    def forward(self, upsample, skip):
        if self.skip_channels is not None and skip.shape[1] != self.skip_channels:
            raise ValueError(f'Number of channels in the skip connection input ({skip.shape[1]}) '
                             f'is different from the expected number of channels ({self.skip_channels})')
        out = self.upconv(upsample)
        out = self.concat(out, skip)
        return out


class CenterCropConcat(nn.Module):
    def forward(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError(f'x and y inputs contain a different number of samples')
        height = min(x.size(2), y.size(2))
        width = min(x.size(3), y.size(3))

        x = self.center_crop(x, height, width)
        y = self.center_crop(y, height, width)

        res = torch.cat([x, y], dim=1)
        return res

    @staticmethod
    def center_crop(x, target_height, target_width):
        current_height = x.size(2)
        current_width = x.size(3)
        min_h = (current_width - target_width) // 2
        min_w = (current_height - target_height) // 2
        return x[:, :, min_w:(min_w + target_height), min_h:(min_h + target_width)]