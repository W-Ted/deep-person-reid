from __future__ import absolute_import
from __future__ import division

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet50_fc512', 'resnet50_angularlinear', 'resnet50_fc512_angular', 'resnet50_arcface', 'resnet50_cosface']

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo

from torch.autograd import Variable
from torch.nn import Parameter
import math


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def l2_norm(input,axis=1):
	norm = torch.norm(input,2,axis,True)
	output = torch.div(input, norm)
	return output

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)

# Arcface head
class Arcface(nn.Module):
	# implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
	def __init__(self, embedding_size=2048, classnum=51332, s=64.0, m=0.1):       # default embedding_size=512, s=64, m=0.5
		super(Arcface, self).__init__()
		self.classnum = classnum
		self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
		# initial kernel
		self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
		self.m = m  # the margin value, default is 0.5
		self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
		self.cos_m = math.cos(m)
		self.sin_m = math.sin(m)
		self.mm = self.sin_m * m  # issue 1
		self.threshold = math.cos(math.pi - m)
		# self.ce_loss = nn.CrossEntropyLoss()

	def forward(self, embbedings):
		# embbedings = l2_norm(embbedings)
		# weights norm
		nB = len(embbedings)
		kernel_norm = l2_norm(self.kernel, axis=0)
		# cos(theta+m)
		cos_theta = torch.mm(embbedings, kernel_norm)
		#         output = torch.mm(embbedings,kernel_norm)
		cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
		cos_theta_2 = torch.pow(cos_theta, 2)
		sin_theta_2 = 1 - cos_theta_2
		sin_theta = torch.sqrt(sin_theta_2)
		cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
		# this condition controls the theta+m should in range [0, pi]
		#      0<=theta+m<=pi
		#     -m<=theta<=pi-m
		cond_v = cos_theta - self.threshold
		cond_mask = cond_v <= 0
		keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
		cos_theta_m[cond_mask] = keep_val[cond_mask]
		# output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
		# idx_ = torch.arange(0, nB, dtype=torch.long)
		# output[idx_, label] = cos_theta_m[idx_, label]
		# output *= self.s  # scale up in order to make softmax work, first introduced in normface
		return cos_theta, cos_theta_m
		# return self.ce_loss(output, label)

# Cosface head
class Am_softmax(nn.Module):
	# implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
	def __init__(self, embedding_size=2048, classnum=51332):         # default embedding_size=512
		super(Am_softmax, self).__init__()
		self.classnum = classnum
		self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
		# initial kernel
		self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
		self.m = 0.0  # 0.35 additive margin recommended by the paper  m=0.35
		self.s = 50.0  # see normface https://arxiv.org/abs/1704.06369  s=30
		#self.ce_loss = CrossEntropyLoss()

	def forward(self, embbedings):
		# embbedings = l2_norm(embbedings)
		kernel_norm = l2_norm(self.kernel, axis=0)
		cos_theta = torch.mm(embbedings, kernel_norm)
		cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
		phi = cos_theta - self.m
		# labell = label             # size=[B]
		# label = label.view(-1, 1)  # size=(B,1)
		# index = cos_theta.data * 0.0  # size=(B,Classnum)
		# index.scatter_(1, label.data.view(-1, 1), 1)
		# index = index.byte()
		# output = cos_theta * 1.0
		# output[index] = phi[index]  # only change the correct predicted output
		# output *= self.s  # scale up in order to make softmax work, first introduced in normface
		return cos_theta, phi
		# print(output.shape, label.shape, labell.shape)
		# return self.ce_loss(output, labell)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Residual network.
    
    Reference:
        He et al. Deep Residual Learning for Image Recognition. CVPR 2016.

    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnet50_fc512``: ResNet50 + FC.
    """
    
    def __init__(self, num_classes, loss, block, layers,
                 last_stride=2,
                 fc_dims=None,
                 angular_linear=None,
                 arc_face=None,
                 cos_face=None,
                 dropout_p=None,
                 **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.loss = loss
        self.feature_dim = 512 * block.expansion
        
        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(fc_dims, 512 * block.expansion, dropout_p)

        if angular_linear:
            self.classifier = AngleLinear(self.feature_dim, num_classes)
        elif arc_face:
            self.classifier = Arcface(embedding_size=self.feature_dim, classnum=num_classes)
        elif cos_face:
            self.classifier = Am_softmax(embedding_size=self.feature_dim, classnum=num_classes)
        else:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.cos_face = cos_face
        self.arc_face = arc_face


        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        
        self.feature_dim = fc_dims[-1]
        
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        
        if self.fc is not None:
            v = self.fc(v)
        
        if not self.training:
            return v

        if self.cos_face or self.arc_face:
            y = self.classifier(l2_norm(v))
        else:
            y = self.classifier(v)
            # 如果self.angular = True, 这里返回的是y是[B,Classnum,2]

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        elif self.loss == 'a-softmax':
            return y
        elif self.loss == 'center':             # resnet50_fc512
            return y, v
        elif self.loss == 'n-pair':
            return v
        elif self.loss == 'ms':
            return l2_norm(v)
        elif self.loss == 'histogram':          # resnet50_fc512
            return v
        elif self.loss == 'center+angular':     # resnet50_fc512_angular
            return y, v
        elif self.loss == 'focal':
            return y
        elif self.loss == 'arcface':
            return y
        elif self.loss == 'cosface':
            return y
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print('Initialized model with pretrained weights from {}'.format(model_url))


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""


def resnet18(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model


def resnet34(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])
    return model


def resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model



def resnet101(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])
    return model


def resnet152(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'])
    return model


"""
resnet + fc
"""
def resnet50_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet50_angularlinear(num_classes, loss='a-softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        angular_linear=True,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def resnet50_fc512_angular(num_classes, loss='center+angular', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        angular_linear=True,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def resnet50_arcface(num_classes, loss='arcface', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        arc_face=True,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def resnet50_cosface(num_classes, loss='cosface', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        cos_face=True,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model