# AUTOGENERATED! DO NOT EDIT! File to edit: ResNet.ipynb (unless otherwise specified).

__all__ = ['get_runner', 'NestedModel', 'TestMixingGrads', 'AutoConv', 'ConvBatch', 'ConvLayer', 'Identity', 'BaseRes',
           'ResBlock', 'XResNet', 'NestedSequentialModel', 'GetResnet']

# Cell
from .basic_operations_01 import *
from .fully_connected_network_02 import *
from .model_training_03 import *
from .convolutions_pooling_04 import *
from .callbacks_05 import *
from .batchnorm_06 import *
from .optimizers_07 import *

# Cell
def get_runner(model=None, layers=None, lf=None, callbacks=[Stats([accuracy]), ProgressCallback(), HyperRecorder(['lr'])], opt=None, db=None):
    "Helper function to get a quick runner"
    if model is None:
        model = SequentialModel(*layers) if layers is not None else get_linear_model(0.1)[0]
    lf = CrossEntropy() if lf is None else lf
    db = db if db is not None else get_mnist_databunch()
    opt = opt if opt is not None else adam_opt()
    learn = Learner(model,lf,opt,db)
    return Runner(learn, callbacks)

# Cell
class NestedModel(Module):
    "NestModel that allows for a sequential model to be called withing an outer model"
    def __init__(self):
        super().__init__()

    def forward(self,xb): return self.layers(xb)

    def bwd(self, out, inp): self.layers.backward()

    def parameters(self):
        for p in self.layers.parameters(): yield p

    def __repr__(self): return f'\nSubModel( \n{self.layers}\n)'

# Cell
class TestMixingGrads(NestedModel):
    "Test module to see if nested SequentialModels will work"
    def __init__(self):
        super().__init__()
        self.layers = SequentialModel(Linear(784, 50, True), ReLU(), Linear(50,25, False))

# Cell
class AutoConv(Conv):
    "Automatic resizing of padding based on kernel size to ensure constant dimensions of input to output"
    def __init__(self, n_in, n_out, kernel_size=3, stride=1):
        padding = Padding(kernel_size // 2)
        super().__init__(n_in, n_out, kernel_size, stride)

# Cell
class ConvBatch(NestedModel):
    "Performs conv then batchnorm"
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, **kwargs):
        self.layers = SequentialModel(AutoConv(n_in, n_out, kernel_size, stride),
                       Batchnorm(n_out))

# Cell
class ConvLayer(NestedModel):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, leak=1, Activation=ReLU):
        "Implements conv, batchnorm, relu pass"
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.layers = SequentialModel(ConvBatch(n_in, n_out, kernel_size, stride),
                           Activation())

    def __repr__(self): return f'ConvBnActivation({self.n_in}, {self.n_out})'

# Cell
class Identity(Module):
    "Module to perform the identity connection (what goes in, comes out)"
    def forward(self,xb): return xb
    def bwd(self,out,inp): inp.g += out.g
    def __repr__(self): return f'Identity Connection'

# Cell
class BaseRes(Module):
    "Resblock Layer: ConvBnActivation layer with a skip connection between inputs and outputs, dynamically changes sizing"
    def __init__(self, expansion, n_in, n_h, stride=1, **kwargs):
        super().__init__()

        n_in, n_out = n_in*expansion, n_h*expansion

        if expansion == 1: layers = [ConvLayer(n_in, n_h, 3, stride=stride), ConvLayer(n_h, n_out, 3, stride=stride)]
        else: layers = [
            ConvLayer(n_in, n_h, 1),
            ConvLayer(n_h, n_h, 3, stride=stride),
            ConvLayer(n_h, n_out, 1)
        ]

        self.conv_layer = SequentialModel(*layers)

        self.identity = Identity() if n_in == n_out else SequentialModel(Pool('Average', ks=2), ConvLayer(n_in, n_out, kernel_size=1))

    def forward(self, xb):
        self.conv_out = self.conv_layer(xb)
        self.id_out = self.identity(xb)
        self.out = self.conv_out + self.id_out
        return self.out

    def bwd(self, out, inp):
        self.conv_out.g = out.g
        self.id_out.g = out.g
        self.conv_layer.backward()
        self.identity.backward()

    def parameters(self):
        for p in self.conv_layer.parameters(): yield p

    def __repr__(self): return f'{self.conv_layer} || {self.identity}'

# Cell
class ResBlock(NestedModel):
    "Adds the final activation after the skip connection addition"
    def __init__(self, expansion, n_in, n_h, stride=1, kernel_size=3, Activation=ReLU, **kwargs):
        super().__init__()
        self.n_in, self.n_h, self.expansion = n_in, n_h, expansion
        self.layers = SequentialModel(BaseRes(expansion, n_in, n_h, kernel_size=kernel_size, stride=stride, **kwargs), Activation())

    def __repr__(self): return f'ResBlock({self.n_in}, {self.n_h*self.expansion})'

# Cell
class XResNet():
    "Class to create ResNet architectures of dynamic sizing"
    @classmethod
    def create(cls, expansion, layers, c_in=3, c_out=1000, size=[28,28]):
        nfs = [c_in, (c_in+1)*8, 64, 64]
        stem = [ConvLayer(nfs[i], nfs[i+1], stride=2 if i==0 else 1)
            for i in range(3)]

        nfs = [64//expansion,64,128,256,512]

        res_layers = [cls._make_layer(expansion, nfs[i], nfs[i+1],
                                      n_blocks=l, stride=1 if i==0 else 2)
                  for i,l in enumerate(layers)]

        res = SequentialModel(
            Reshape(c_in, *size),
            *stem,
            Pool(max_pool, ks=3, stride=2, padding=Padding(1)),
            *res_layers,
            Pool(avg_pool,ks=1), Flatten(),
            Linear(nfs[-1]*expansion, c_out, False),
        )

        return res

    @staticmethod
    def _make_layer(expansion, ni, nf, n_blocks, stride):
        return NestedSequentialModel(*[ResBlock(expansion, ni if i==0 else nf, nf, stride if i==0 else 1)
              for i in range(n_blocks)])

# Cell
#hide
class NestedSequentialModel(SequentialModel):
    def __repr__(self): return f'(\n{super().__repr__()}\n)'

# Cell
def GetResnet(size, **kwargs):
    "Helper function to get ResNet architectures of different sizes"
    if size == 18: return XResNet.create(1, [2,2,2,2], **kwargs)
    elif size == 34: return XResNet.create(1, [3,4,6,3], **kwargs)
    elif size == 50: return XResNet.create(4, [3,4,6,3], **kwargs)
    elif size == 150: return XResNet.create(4, [3,4,23,3], **kwargs)
    elif size == 152: return XResNet.create(4, [3,8,36,3], **kwargs)