# AUTOGENERATED! DO NOT EDIT! File to edit: AdvancedTrainingLoop.ipynb (unless otherwise specified).

__all__ = ['Dataset', 'DataLoader', 'Databunch', 'Runner', 'Callback', 'TrainEvalCallback', 'Stat', 'StatTracker',
           'Stats']

# Cell
from .basic_operations_01 import *
from .fully_connected_network_02 import *
from .training_loop_03 import *
from .convolutions_pooling_04 import *

# Cell
import math

class Dataset():
    def __init__(self, x, y): self.x, self.y = x, y
    def __getitem__(self, i): return self.x[i], self.y[i]
    def __len__(self): return len(self.x)
    def __repr__(self): return f'X: {self.x.shape}, Y: {self.y.shape}'

class DataLoader():
    def __init__(self, ds, batcher, collate_fcn): self.ds, self.batcher, self.collate_fcn = ds, batcher, collate_fcn
    def __iter__(self):
        for b in self.batcher: yield self.collate_fcn([self.ds[i] for i in b])
    @property
    def dataset(self): return self.ds
    def __len__(self): return math.ceil(len(self.ds) / self.batcher.bs)
    def __repr__(self): return f'Data: {self.ds}, bs = {self.batcher.bs}'

class Databunch():
    def __init__(self, train_dl, valid_dl): self.train, self.valid = train_dl, valid_dl

    @property
    def train_ds(self): return self.train.dataset

    @property
    def valid_ds(self): return self.valid.dataset

    def __repr__(self): return f'Databunch(\nTrain: {self.train}, \nValid{self.valid}\n)'

# Cell
class Runner():
    def __init__(self, learner, cbs=None):
        cbs = [] if cbs is None else cbs
        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

        for cb in self.cbs:
            cb.runner = self

        self.learner = learner

    @property
    def model(self): return self.learner.model
    @property
    def optimizer(self): return self.learner.optimizer
    @property
    def loss_func(self): return self.learner.loss_func
    @property
    def databunch(self): return self.learner.db

    def do_one_batch(self, xb, yb):
        self.xb, self.yb = xb, yb

        self.pred = self.learner.model(xb)
        self.loss = self.learner.loss_func(self.pred, yb)
        if self.check_callbacks('after_loss') or not self.learner.model.training: return

        self.learner.loss_func.backward()
        if self.check_callbacks('after_loss_back'): return

        self.learner.model.backward()
        if self.check_callbacks('after_model_back'): return

        self.opt.step()
        if self.check_callbacks('after_opt'): return

        self.opt.zero_grad()
        if self.check_callbacks('after_zero_grad'): return

    def do_all_batches(self, dl):
        self.iters, self.iters_done = len(dl), 0
        for xb, yb in dl:
            if self.stop: break
            if self.check_callbacks('before_batch'): return
            self.do_one_batch(xb,yb)
            if self.check_callbacks('after_batch'): return
        self.iters = 0

        self.stop = False

    def fit(self, epochs, lr=0.1):
        self.lr, self.epochs = lr, epochs
        if self.check_callbacks('before_fit'): return

        for epoch in range(epochs):
            self.epoch = epoch
            if self.check_callbacks('before_epoch'): return
            if not self.check_callbacks('before_train'): self.do_all_batches(self.learner.db.train)
            if not self.check_callbacks('before_valid'): self.do_all_batches(self.learner.db.valid)
            if self.check_callbacks('after_epoch'): break

        if self.check_callbacks('after_fit'): return

    def check_callbacks(self, state):
        for cb in sorted(self.cbs, key=lambda x: x._order):
            f = getattr(cb, state, None)
            if f and f(): return True
        return False

# Cell
class Callback():
    _order = 0
    def __getattr__(self,k):
        #If callback doesn't have an attribute, check the runner
        return getattr(self.runner, k)

    def __repr__(self): return f'{self.__class__.__name__}'

class TrainEvalCallback(Callback):
    _order = 10

    def before_fit(self):
        self.runner.opt = self.learner.optimizer(self.learner.model.parameters(), self.lr)
        self.runner.epochs_done = 0.

    def before_batch(self):
        self.runner.iters_done += 1
        self.runner.epochs_done += 1/self.iters

    def before_valid(self):
        self.model.training = False

    def before_train(self):
        self.model.training = True

    def after_epoch(self):
        self.runner.iters_done = 0

# Cell
class Stat():
    def __init__(self, calc): self.calc, self.value, self.count = calc, 0., 0

    def __call__(self, bs, *args):
        self.value += self.calc(*args) * bs
        self.count += bs

    def reset(self): self.value, self.count = 0., 0

    def __repr__(self): return f'{(self.calc.__name__).capitalize()}: {self.value / self.count}' if self.count > 0 else f'{(self.calc.__name__).capitalize()}'

class StatTracker():
    def __init__(self, metrics, in_train):
        self.in_train = in_train
        self.metrics = [Stat(m) for m in metrics]

    def reset(self):
        self.count, self.tot_loss = 0., 0.
        for met in self.metrics: met.reset()

    def __len__(self): return len(self.metrics)

    def accumulate(self, run):
        bs = run.xb.shape[0]
        self.tot_loss = run.loss * bs
        self.count += bs
        for i,met in enumerate(self.metrics):
            met(bs, run.pred, run.yb)

    def __repr__(self):
        if self.count < 1: return ""
        else:
            printed_stats = f'Loss: {self.tot_loss / self.count}'
            for met in self.metrics:
                printed_stats += f', {met}'
            return f'{"Train" if self.in_train else "Valid"}: {printed_stats}'

class Stats(Callback):
    def __init__(self, metrics):
        self.train, self.valid = StatTracker(metrics, True), StatTracker(metrics, False)

    def before_epoch(self):
        self.train.reset()
        self.valid.reset()

    def after_loss(self):
        stats = self.train if self.model.training else self.valid
        stats.accumulate(self.runner)

    def after_epoch(self):
        print(f'Epoch: {self.epoch+1}')
        print(self.train)
        print(self.valid)