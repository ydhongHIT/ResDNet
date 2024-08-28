import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


def convNxN(in_planes, out_planes, kernel_size=3, stride=1, padding=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    if dilation > 1:
        padding = dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)

class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sqeeze = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
        )
        self.expand = nn.Sequential(
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        sqeeze_relu = self.sqeeze(y)
        y = self.expand(sqeeze_relu).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNetBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

class BasicBlock(nn.Sequential):
    def __init__(self, in_planes, growth_rate, memory_efficient, groups=1, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.add_module('norm1', norm_layer(in_planes)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', convNxN(in_planes, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           groups=groups, dilation=dilation)),
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        return bottleneck_output


class BottleneckBlock(nn.Sequential):
    def __init__(self, in_planes, growth_rate, memory_efficient, groups=1, dilation=1, norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.add_module('norm1', norm_layer(in_planes)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_planes,
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', norm_layer(growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', convNxN(growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           groups=groups, dilation=dilation)),
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        return new_features


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=None, se=True, se_reduction=16):
        super(TransitionBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.bn1 = norm_layer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = norm_layer(out_planes)
        self.se = se
        if se:
            self.se = SELayer(out_planes, se_reduction)

    def forward(self, x):
        out = self.bn2(self.conv1(self.relu(self.bn1(x))))
        if self.se:
            out = self.se(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, block, nb_layers, in_planes, growth_rate, memory_efficient, groups,
                                     slide_windows, dilation, norm_layer):
        super(DenseBlock, self).__init__()
        self.growth_rate = growth_rate
        self.slide_windows = slide_windows
        if slide_windows:
            self.layers = nn.ModuleList(
                [block(in_planes, growth_rate, memory_efficient, groups, dilation=dilation, norm_layer=None, 
                                 ) for _ in range(nb_layers)])
        else:
            self.layers = nn.ModuleList(
                [block(in_planes + i * growth_rate, growth_rate, memory_efficient, groups, dilation=dilation, 
                       norm_layer=None) for i in range(nb_layers)])
      
    def forward(self, x):
        if self.slide_windows:
            x_list = list(torch.split(x, self.growth_rate, dim=1))
            temp = len(x_list)
            for i, layer in enumerate(self.layers):
                y = layer(*x_list[-temp:])
                x_list.append(y)
            return torch.cat(x_list, 1)
        else:
            xx = x
            xcat = [xx]
            for i, layer in enumerate(self.layers):
                y = layer(*xcat)
                xcat.append(y)
            return torch.cat(xcat, 1)

class BasicStem(nn.Module):
    def __init__(self, growth_rate, width, memory_efficient, norm_layer=None):
        super(BasicStem, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn2 = norm_layer(64)
        self.conv2 = nn.Conv2d(64, width * growth_rate, kernel_size=1, stride=1, padding=0,
                                   bias=False)

    def forward(self, x):

        out = self.conv2(self.relu(self.bn2((self.maxpool(self.relu(self.bn1(self.conv1(x))))))))

        return out

class BottleStem(nn.Module):
    def __init__(self, growth_rate, width, memory_efficient, norm_layer=None):
        super(BottleStem, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.bn2 = norm_layer(64)
        self.conv3 = nn.Conv2d(64, width * growth_rate, kernel_size=1, stride=1, padding=0,
                                   bias=False)

    def forward(self, x):

        out = self.conv3(self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))))

        return out
    
class ResDNetBlock(ResNetBlockBase):
    def __init__(self, block, nb_layers, in_channels, out_channels, growth_rate, memory_efficient, groups, dilation, 
                 norm_layer=None, stride=1, slide_windows=True, se=False):
        super().__init__(in_channels, out_channels, stride)
        self.downsample = False
        self.se = se
        if stride != 1:
            self.downsample = True
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.denseblock = DenseBlock(block, nb_layers, in_channels, growth_rate, memory_efficient, groups,
                                     slide_windows, dilation, norm_layer)
        self.transition = TransitionBlock(in_channels + nb_layers * growth_rate, out_channels, norm_layer)
        if self.se:
            self.se_block = SELayer(out_channels, 16)
      
    def forward(self, x):
        if self.downsample:
            x = self.pool(x)
        residual = x
        x = self.transition(self.denseblock(x))
        if self.se:
            x = self.se_block(x)
        if x.size()[1] != residual.size()[1]:
            x = x + F.pad(residual, [0, 0, 0, 0, 0, x.size()[1] - residual.size()[1], 0, 0])
        else:
            x = x + residual
        return x

class ResDNet(nn.Module):
    def __init__(self, dense_type, nb_layers, growth_rate, memory_efficient, se, groups, layers,             
                 norm_layer=None, num_classes=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(ResDNet, self).__init__()

        if dense_type == "B":
            densestype = BottleneckBlock
        elif dense_type == "A":
            densestype = BasicBlock
        else:
            RuntimeError('dense type error')

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.num_classes = num_classes

        self.stem = BasicStem(growth_rate[0], nb_layers, memory_efficient, norm_layer)

        self.layer1 = self._make_resdnet_layer(densestype, nb_layers, growth_rate[0], memory_efficient, se, groups, ResDNetBlock, norm_layer, 
                                     nb_layers * growth_rate[0], layers[0])
        self.layer2 = self._make_resdnet_layer(densestype, nb_layers, growth_rate[1], memory_efficient, se, groups, ResDNetBlock, norm_layer, 
                                     nb_layers * growth_rate[1], layers[1], stride=2)
        self.layer3 = self._make_resdnet_layer(densestype, nb_layers, growth_rate[2], memory_efficient, se, groups, ResDNetBlock, norm_layer, 
                                     nb_layers * growth_rate[2], layers[2], stride=2)
        self.layer4 = self._make_resdnet_layer(densestype, nb_layers, growth_rate[3], memory_efficient, se, groups, ResDNetBlock, norm_layer, 
                                     nb_layers * growth_rate[3], layers[3], stride=2, fixed_output=True)


        if num_classes is not None:
            curr_channels = 2048
            self.in_planes = curr_channels
            self.bn = norm_layer(curr_channels)
            self.relu = nn.ReLU(inplace=True)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_resdnet_layer(self, denseblock, nb_layers, growth_rate, memory_efficient, se, groups, block, norm_layer, planes, blocks, dilation=1, stride=1, fixed_output=False):

        layers = []

        if blocks == 1:
            if fixed_output == True:
                layers.append(block(denseblock, nb_layers, planes, 2048, growth_rate, memory_efficient, groups, dilation, 
                          norm_layer, stride=stride, slide_windows=True, se=se))
            else:
                layers.append(block(denseblock, nb_layers, planes, 2 * planes, growth_rate, memory_efficient, groups, dilation, 
                          norm_layer, stride=stride, slide_windows=True, se=se))
        else:
            layers.append(block(denseblock, nb_layers, planes, planes, growth_rate, memory_efficient, groups, dilation, 
                  norm_layer, stride=stride, slide_windows=True, se=se))
            for i in range(1, blocks):
                if i != (blocks-1):
                    layers.append(block(denseblock, nb_layers, planes, planes, growth_rate, memory_efficient, groups, dilation, 
                          norm_layer, slide_windows=True, se=se))
                else:
                    if fixed_output == True:
                        layers.append(block(denseblock, nb_layers, planes, 2048, growth_rate, memory_efficient, groups, dilation, 
                          norm_layer, slide_windows=True, se=se))
                    else:
                        layers.append(block(denseblock, nb_layers, planes, 2 * planes, growth_rate, memory_efficient, groups, dilation, 
                          norm_layer, slide_windows=True, se=se))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.num_classes is not None:
            x = self.avgpool(self.relu(self.bn(x)))
            x = x.view(-1, self.in_planes)
            x = self.linear(x)
        return x

def ResDNet-B-129():
    model = ResDNet(dense_type="B", nb_layers=4, growth_rate=[32, 64, 128, 256], memory_efficient=True, se=False, groups=1, layers=[2, 4, 6, 2])
    return model

def ResDNet-B-SE-129():
    model = ResDNet(dense_type="B", nb_layers=4, growth_rate=[32, 64, 128, 256], memory_efficient=True, se=True, groups=1, layers=[2, 4, 6, 2])
    return model








