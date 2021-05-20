# AUTOGENERATED! DO NOT EDIT! File to edit: cycleGAN.ipynb (unless otherwise specified).

__all__ = ['AutoTransConv', 'trans_conv_norm_relu', 'pad_conv_norm_relu', 'conv_norm_relu', 'ResBlock', 'generator',
           'discriminator', 'cycleGAN', 'DynamicLoss', 'CycleGANLoss', 'cycleGANTrainer']

# Cell
from ModernArchitecturesFromPyTorch.nb_XResNet import *
from nbdev.showdoc import show_doc

# Cell
class AutoTransConv(nn.Module):
    "Automatic padding of transpose convolution for input-output feature size"
    def __init__(self, n_in, n_out, ks, stride, bias=True):
        super().__init__()
        padding = ks // 2
        self.conv = nn.ConvTranspose2d(n_in, n_out, ks, stride, padding=padding, output_padding=padding, bias=bias)

# Cell
def trans_conv_norm_relu(n_in, n_out, norm_layer, bias, ks=3, stride=2):
    "Transpose convolutional layer"
    return [AutoTransConv(n_in, n_out, ks=ks, stride=stride, bias=bias),
                         norm_layer(n_out),
                         nn.ReLU()
                         ]

# Cell
def pad_conv_norm_relu(n_in, n_out, norm_layer, padding_mode="zeros", pad=1, ks=3, stride=1, activ=True, bias=True):
    "Adding ability to specify different paddings to convolutional layer"
    layers = []
    if padding_mode != "zeros":
        pad = 0
        if padding_mode == "reflection": layers.append(nn.ReflectionPad2d(pad))
        elif padding_mode == "border": layers.append(nn.ReplicationPad2d(pad))

    layers.append(AutoConv(n_in, n_out, ks, stride=stride, padding_mode=padding_mode, bias=bias))
    layers.append(norm_layer(n_out))

    if activ: layers.append(nn.ReLU(inplace=True))

    return layers

# Cell
def conv_norm_relu(n_in, n_out, norm_layer=None, ks=3, bias:bool=True, pad=1, stride=1, activ=True, a=0.2):
    "Convolutional layer"
    layers = []
    layers.append(nn.Conv2d(n_in, n_out, ks, stride=stride, padding=pad))
    if norm_layer != None: layers.append(norm_layer(n_out))
    if activ: layers.append(nn.LeakyReLU(a, True))
    return nn.Sequential(*layers)

# Cell
class ResBlock(nn.Module):
    def __init__(self, dim, padding_mode, bias, dropout, norm_layer=nn.InstanceNorm2d):
        "Residual connections for middle section of generator"
        super().__init__()
        layers = []
        layers += pad_conv_norm_relu(dim, dim, norm_layer, padding_mode, bias=bias)
        if dropout > 0: layers.append(nn.Dropout(dropout))
        layers += (pad_conv_norm_relu(dim, dim, norm_layer, padding_mode, bias=bias, activ=False))
        self.xb = nn.Sequential(*layers)

    def forward(self, xb): return xb + self.conv(xb)

# Cell
def generator(n_in, n_out, n_f=64, norm_layer=None, dropout=0., n_blocks=6, pad_mode="reflection"):
    "Generator that maps an input of one domain to the other"
    norm_layer = norm_layer if norm_layer is not None else nn.InstanceNorm2d
    bias = (norm_layer == nn.InstanceNorm2d)

    layers = []

    layers += pad_conv_norm_relu(n_in, n_f, norm_layer, pad_mode, pad=3, ks=7, bias=bias)
    for i in range(2):
        layers += pad_conv_norm_relu(n_f, n_f*2, norm_layer, 'zeros', stride=2, bias=bias)
        n_f*=2

    layers += [ResBlock(n_f, pad_mode, bias, dropout, norm_layer) for _ in range(n_blocks)]

    for i in range(2):
        layers += trans_conv_norm_relu(n_f, n_f//2, norm_layer, bias=bias)
        n_f //= 2

    layers.append(nn.ReflectionPad2d(3))
    layers.append(nn.Conv2d(n_f, n_out, kernel_size=7, padding=0))
    layers.append(nn.Tanh())

    return nn.Sequential(*layers)

# Cell
def discriminator(c_in, n_f, n_layers, norm_layer=None, sigmoid=False):
    "Discrminator to classify input as belonging to one class or the other"
    norm_layer = nn.InstanceNorm2d if norm_layer is None else norm_layer
    bias = (norm_layer == nn.InstanceNorm2d)
    layers = []
    layers += (conv_norm_relu(c_in, n_f, ks=4, stride=2, pad=1))

    for i in range(n_layers-1):
        new_f = 2*n_f if i <= 3 else n_f
        layers += (conv_norm_relu(n_f, new_f, norm_layer, ks=4, stride=2, pad=1, bias=bias))
        n_f = new_f

    new_f = 2*n_f if n_layers <= 3 else n_f

    layers += (conv_norm_relu(n_f, new_f, norm_layer, ks=4, stride=1, pad=1, bias=bias))
    layers.append(nn.Conv2d(new_f, 1, kernel_size=4, stride=1, padding=1))
    if sigmoid: layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

# Cell
class cycleGAN(nn.Module):
    def __init__(self, c_in, c_out, n_f=64, disc_layers=3, gen_blocks=6, drop=0., norm_layer=None, sigmoid=False):
        super().__init__()
        self.a_discriminator = discriminator(c_in, n_f, disc_layers, norm_layer, sigmoid)
        self.b_discriminator = discriminator(c_in, n_f, disc_layers, norm_layer, sigmoid)
        self.generate_a = generator(c_in, c_out, n_f, norm_layer, drop, gen_blocks)
        self.generate_b = generator(c_in, c_out, n_f, norm_layer, drop, gen_blocks)

    def forward(self, real_A, real_B):
        generated_a, generated_B = self.generate_a(real_B), self.generate_b(real_A)
        if not self.training: return generated_a, generated_b
        id_a, id_b = self.generate_a(real_A), self.generate_b(real_B)
        return [generated_a, generated_b, id_a, id_b]

# Cell
class DynamicLoss(nn.Module):
    def __init__(self, loss_fn):
        "Loss allowing for dynamic resizing of prediction based on shape of output"
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, targ, **kwargs):
        targ = output.new_ones(*pred.shape) if targ == 1 else output.new_zeros(*pred.shape)
        return self.loss_fn(pred, targ, **kwargs)

# Cell
class CycleGANLoss(nn.Module):
    def __init__(self, model, loss_fn=F.mse_loss, la=10., lb=10, lid=0.5):
        "CycleGAN loss"
        super().__init__()
        self.model,self.la,self.lb,self.lid = model,la,lb,lid
        self.loss_fn = DynamicLoss(loss_fn)

    def store_inputs(self, inputs):
        self.reala,self.realb = inputs

    def forward(self, pred, target):
        gena, genb, ida, idb = pred

        self.id_loss = self.lid * (self.la * F.l1_loss(ida, self.reala) + self.lb * F.l1_loss(idb,self.realb))

        self.gen_loss = self.crit(self.model.a_discriminator(gena), True) + self.crit(self.model.b_discriminator(genb), True)

        self.cyc_loss = self.la*  F.l1_loss(self.model.generate_a(genb), self.reala) + self.lb*F.l1_loss(self.model.generate_b(gena), self.realb)

        return self.id_loss+ self.gen_loss + self.cyc_loss

# Cell
class cycleGANTrainer(Callback):
    "Trainer to sequence timing of training both the discriminator as well as the generator's"
    _order = -20

    def set_grad(self, da=False, db=False):
        in_gen = (not da) and (not db)
        requires_grad(self.learn.model.generate_a, in_gen)
        requires_grad(self.learn.generate_b, in_gen)
        requires_grad(self.learn.a_discriminator, da)
        requires_grad(self.learn.b_discriminator, db)
        if not gen:
            self.opt_D_A.lr, self.opt_D_A.mom = self.learn.opt.lr, self.learn.opt.mom
            self.opt_D_A.wd, self.opt_D_A.beta = self.learn.opt.wd, self.learn.opt.beta
            self.opt_D_B.lr, self.opt_D_B.mom = self.learn.opt.lr, self.learn.opt.mom
            self.opt_D_B.wd, self.opt_D_B.beta = self.learn.opt.wd, self.learn.opt.beta

    def before_fit(self, **kwargs):
        self.ga = self.learn.model.generate_a
        self.gb = self.learn.generate_b
        self.da = self.learn.a_discriminator
        self.db = self.learn.b_discriminator
        self.loss_fn = self.learn.loss_func.loss_func

        if not getattr(self,'opt_gen',None):
            self.opt_gen = self.learn.opt.new([nn.Sequential(*flatten_model(self.ga), *flatten_model(self.gb))])
        else:
            self.opt_gen.lr,self.opt_gen.wd = self.opt.lr,self.opt.wd
            self.opt_gen.mom,self.opt_gen.beta = self.opt.mom,self.opt.beta

        if not getattr(self,'opt_da',None):
            self.opt_da = self.learn.opt.new([nn.Sequential(*flatten_model(self.da))])

        if not getattr(self,'opt_db',None):
            self.opt_db = self.learn.opt.new([nn.Sequential(*flatten_model(self.db))])

        self.learn.opt.opt = self.opt_gen
        self.set_grad()

    def before_batch(self, last_input, **kwargs):
        self.learn.loss_func.store_inputs(last_input)

    def after_batch(self, last_input, last_output, **kwags):
        #Discriminator loss
        self.ga.zero_grad(), self.gb.zero_grad()
        fakea, fakeb = last_output[0].detach(), last_output[1].detach()

        reala,realb = last_input
        self.set_grad(da=True)
        self.da.zero_grad()
        lossda = 0.5 * (self.loss_fn(self.da(reala), True) + self.loss_fn(self.da(fakea), False))
        lossda.backward()
        self.opt_da.step()

        self.set_grad(db=True)
        self.opt_db.zero_grad()
        lossdb = 0.5 * (self.loss_fn(self.db(realb), True) + self.loss_fn(self.da(fakeb), False))
        lossdb.backward()
        self.opt_db.step()

        self.set_grad()