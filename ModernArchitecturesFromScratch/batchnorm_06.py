# AUTOGENERATED! DO NOT EDIT! File to edit: Batchnorm.ipynb (unless otherwise specified).

__all__ = ['CheckGrad', 'Batchnorm', 'get_conv_model', 'get_conv_learner', 'get_conv_runner']

# Cell
from .basic_operations_01 import *
from .fully_connected_network_02 import *
from .model_training_03 import *
from .convolutions_pooling_04 import *
from .callbacks_05 import *

# Cell
class CheckGrad(Callback):
    _order = 100
    def require_grad(self):
        #pdb.set_trace()
        for p in self.runner.model.parameters():
            p.d.requires_grad_(True)

    def after_loss(self):
        #pdb.set_trace()
        if self.iters_done < 2:
            #run.check1 = [self.runner.xb, self.runner.yb, self.pred] + [m.d.clone() for m in self.model.parameters()]
            return
        else:
            #run.check2 = [self.runner.xb, self.runner.yb, self.pred] + [m.d.clone() for m in self.model.parameters()]
            self.loss.backward()
            self.runner.pytorch_gradients = [m.d.grad for m in self.runner.model.parameters()]
            return True

    def after_model_back(self):
        #pdb.set_trace()
        return True

    def before_batch(self):
        #pdb.set_trace()
        if self.iters_done < 2:
            self.copy_batch_x = self.runner.xb.clone()
            self.copy_batch_y = self.runner.yb.clone()
        else:
            self.runner.xb = self.copy_batch_x
            self.runner.yb = self.copy_batch_y
            self.runner.xb.requires_grad_(True)
            return

    def after_batch(self):
        #pdb.set_trace()
        if self.iters_done < 2:
            self.runner.custom_gradients = [m.grad.clone() for m in self.runner.model.parameters()]
            self.require_grad()
            return
        else:
            return True

    def before_valid(self): return True
    def after_epoch(self): return True
    def after_fit(self):
        #for i in range(len(run.check1)):
            #test_near(run.check1[i], run.check2[i])
        for i in range(len(self.runner.custom_gradients)):
            test_near(self.runner.pytorch_gradients[i], self.runner.custom_gradients[i])

# Cell
class Batchnorm(Module):
    "Module for applying batch normalization"
    def __init__(self, nf, mom=0.1, eps=1e-6):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.multiplier = Parameter(torch.ones(1,nf, 1, 1))
        self.adder = Parameter(torch.zeros(1,nf,1,1))
        self.means = torch.zeros(1,nf,1,1)
        self.vars = torch.ones(1,nf,1,1)

    def update(self, xb):
        #Get the mean and standard deviation of the batch, update running average
        mean = xb.mean(dim=(0,2,3), keepdim=True)
        var = xb.std(dim=(0,2,3), keepdim=True)
        self.mean = self.mom * self.means + (1-self.mom) * mean
        self.vars = self.mom * self.vars + (1-self.mom) * var
        return mean, var


    def forward(self, xb):
        mean, var = self.update(xb)
        self.after_stats = (xb - mean) / (var + self.eps).sqrt()
        self.after_scaling = self.after_stats * self.multiplier.d + self.adder.d
        return self.after_scaling

    def bwd(self, out, inp):
        bs = out.g.shape[0]

        self.multiplier.update((out.g * self.after_stats).sum(dim=(0,2,3), keepdim=True))
        self.adder.update(out.g.sum(dim=(0,2,3), keepdim=True))

        var_factor = 1./(self.vars+self.eps).sqrt()
        mean_factor = inp - self.means

        delta_norm = out.g * self.multiplier.d

        delta_var = delta_norm * mean_factor * -0.5 * (self.vars + self.eps)**(-3/2)
        delta_mean = delta_norm * -var_factor + delta_var * 1 / bs * -2 * mean_factor

        inp.g = (delta_norm * var_factor) + (delta_mean / bs) + (delta_var * 2 / bs * mean_factor)

    def __repr__(self): return f'Batchnorm'

# Cell
def get_conv_model():
    "Returns a sequential convolutional model"
    return SequentialModel(Reshape(1, 28, 28),
            Conv(1, 8, stride=2),
            Batchnorm(8),
            Flatten(),
            Linear(1352, 10, False)
    )

def get_conv_learner():
    "Returns a conv learner object"
    m = get_conv_model()
    o = Optimizer
    l = CrossEntropy()
    db = Databunch(*get_small_datasets())
    return Learner(m,l,o,db)

def get_conv_runner(callbacks):
    "Returns a convolutionary model runner, ready to be fitted with given `callbacks`"
    learn = get_conv_learner()
    run = Runner(learn, callbacks)
    return run