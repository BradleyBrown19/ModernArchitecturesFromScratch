
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/ScratchToPytorch.ipynb
import torch
import torch.nn as nn
import fastai
from fastai import datasets
from fastai.vision import *
import pdb
import time
import fire
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time
from ModernArchitecturesFromScratch.callbacks_05 import *

def get_imaggenette():
    path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)
    tfms = get_transforms()
    return (ImageList.from_folder(path).split_by_rand_pct(0.2).label_from_folder().transform(tfms, size=224).databunch())

class CustomLearner():
    def __init__(self, model, loss_func, optimizer, db):
        "Wrapper for model, loss function, optimizer and databunch"
        self.model, self.loss_func, self.optimizer, self.db = model, loss_func, optimizer(model.parameters()), db

def get_learner(model, data=get_imaggenette(), loss=nn.CrossEntropyLoss(), optimizer=optim.Adam):
    return CustomLearner(model, loss, optimizer, data)

def get_runner(learn, callbacks:List): return Runner(learn, callbacks)

class ProgressCallback(Callback):
    "Callback to make a nice progress bar with metrics for training. Slightly modified version of: https://github.com/fastai/course-v3/blob/master/nbs/dl2/09c_add_progress_bar.ipynb"
    _order=-1
    def before_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.runner.logger = partial(self.mbar.write, table=True)

    def after_fit(self): self.mbar.on_iter_end()
    def after_batch(self): self.pb.update(self.iters_done)
    def before_train(self): self.set_pb(self.runner.databunch.train_dl)
    def before_valid(self): self.set_pb(self.runner.databunch.valid_dl)

    def set_pb(self, dl):
        self.pb = progress_bar(dl, parent=self.mbar)
        self.mbar.update(self.epoch)
        self.pb.update(0)

class TrainEvalCallback(Callback):
    "Keeps track of training/eval mode of model and progress through training"
    _order = 10

    def before_fit(self):
        #self.runner.opt = self.learner.optimizer(self.learner.model.parameters(), self.lr)
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

def accuracy(out, yb, *args): return (torch.argmax(out, dim=1)==yb).float().mean()

class CudaCallback(Callback):
    def before_fit(self): self.model.cuda()

class Runner():
    "All encompossing class to train a model with specific callbacks"
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
    @property
    def dl(self): return self.learner.db.train_dl if self.model.training else self.learner.db.valid_dl

    def do_one_batch(self, xb, yb):
        "Applies forward and backward passes of model to one batch"

        self.pred = self.learner.model(xb)

        self.loss = self.learner.loss_func(self.pred, yb)
        if self.check_callbacks('after_loss') or not self.learner.model.training: return

        self.loss.backward()
        if self.check_callbacks('after_loss_back'): return

        self.optimizer.step()
        if self.check_callbacks('after_opt'): return

        self.optimizer.zero_grad()
        if self.check_callbacks('after_zero_grad'): return

    def do_all_batches(self, dl):
        "Runs every batch of a dataloader through `do_one_batch`"
        self.iters, self.iters_done = len(dl), 0
        for xb, yb in dl:
            if self.stop: break
            self.xb, self.yb = xb,yb
            if self.check_callbacks('before_batch'): return
            self.do_one_batch(self.xb,self.yb)
            if self.check_callbacks('after_batch'): return
        self.iters = 0

        self.stop = False

    def fit(self, epochs, lr=0.1):
        "Method to fit the model `epoch` times using learning rate `lr`"
        self.optimizer.lr, self.epochs = lr, epochs
        if self.check_callbacks('before_fit'): return

        for epoch in range(epochs):
            self.epoch = epoch
            if self.check_callbacks('before_epoch'): return

            if not self.check_callbacks('before_train'): self.do_all_batches(self.learner.db.train_dl)

            with torch.no_grad():
                if not self.check_callbacks('before_valid'): self.do_all_batches(self.learner.db.valid_dl)

            if self.check_callbacks('after_epoch'): break

        if self.check_callbacks('after_fit'): return

    def check_callbacks(self, state):
        "Helper functions to run through each callback, calling it's state method if applicable"
        for cb in sorted(self.cbs, key=lambda x: x._order):
            f = getattr(cb, state, None)
            if f and f(): return True
        return False