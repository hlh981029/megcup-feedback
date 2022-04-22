import os
from megengine import module as nn
from .operators import BaseOperator, Conv2dOp
from .searcher_utils import write_to_json, load_config
import logging
logger = logging.getLogger('megcup')


def wrap_model(model, config, search_op='Conv2d', init_rates=None):
    op = 'nn.' + search_op
    for name, module in model.named_children():
        if isinstance(module, eval(op)):
            if module.kernel_size[0] > 1 and module.kernel_size[0] < 7:
                moduleWrap = eval(search_op + 'Op')(module, init_rates, config, config['S'])
                logger.info('Wrap model %s to %s.' % (str(module), str(moduleWrap)))
                setattr(model, name, moduleWrap)
        elif isinstance(module, BaseOperator):
            pass
        else:
            wrap_model(module, config, search_op, init_rates)


def set_model(model, config, search_op='Conv2d', init_rates=None, prefix=""):
    config['model'] = load_config(config['search_log'])
    op = 'nn.' + search_op
    for name, module in model.named_children():
        if prefix == "":
            fullname = name
        else:
            fullname = prefix + '.' + name
        if isinstance(module, eval(op)):
            if module.kernel_size[0] > 1 and module.kernel_size[0] < 7:
                if isinstance(config["model"][fullname], int):
                    config['model'][fullname] = (config['model'][fullname], )
                module.dilation = (
                    config['model'][fullname][0],
                    config['model'][fullname][0]
                )
                module.padding = (
                    config["model"][fullname][0] * (module.kernel_size[0] - 1) // 2,
                    config["model"][fullname][0] * (module.kernel_size[0] - 1) // 2,
                )
                setattr(model, name, module)
                logger.info('Set module %s dilation as: [%d]' % (fullname, module.dilation[0]))
        elif isinstance(module, BaseOperator):
            pass
        else:
            set_model(module, config, search_op, init_rates, fullname)


def update_ema_model(model, ema_model, step, config):
    if config['finetune']:
        return
    if step % config['search_interval'] == 0 \
            and step < config['max_step']:
        for (name, module), (ema_name, ema_module) in zip(model.named_children(), ema_model.named_children()):
            if isinstance(module, BaseOperator):
                ema_module.weights = module.weights
                ema_module.rates = module.rates
                logger.info(f'Set EMA module {ema_name} dilation as: {ema_module.rates}')
            else:
                update_ema_model(module, ema_module, step, config)


def print_ema_model(ema_model, step, config):
    if config['finetune']:
        return
    if step % config['search_interval'] == 0 \
            and step < config['max_step']:
        for ema_name, ema_module in ema_model.named_children():
            if isinstance(ema_module, BaseOperator):
                print(ema_module.rates, ema_module.weights)
            else:
                print_ema_model(ema_module, step, config)


def print_model(model):

    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            print(name, module.dilation, module.padding)
        else:
            print_model(module)


class Searcher():
    def __init__(self, search_config, model, logdir):
        search_config['model'] = {}
        self.config = search_config
        self.model = model
        self.logdir = logdir
        self.S = self.config['S']
        os.makedirs(self.logdir, exist_ok=True)

    def step(self):
        if self.config['finetune']:
            return
        self.config['step'] += 1
        if (self.config['step']) % self.config['search_interval'] == 0 \
                and (self.config['step']) < self.config['max_step']:
            self.search()
            for name, module in self.model.named_modules():
                if isinstance(module, BaseOperator):
                    self.config['model'][name] = module.op_layer.dilation
            write_to_json(self.config, os.path.join(self.logdir, 'local_search_config_step%d.json' % self.config['step']))
        elif (self.config['step'] + 1) == self.config['max_step']:
            self.search_estimate_only()

    def get_step(self):
        return self.config['step']

    def search(self):
        for _, module in self.model.named_modules():
            if isinstance(module, BaseOperator):
                module.estimate()
                module.expand()

    def search_estimate_only(self):
        for _, module in self.model.named_modules():
            if isinstance(module, BaseOperator):
                module.estimate()

    def wrap_model(self, model, config, search_op='Conv2d', init_rates=None):
        op = 'nn.' + search_op
        for name, module in model.named_children():
            if isinstance(module, eval(op)):
                if module.kernel_size[0] > 1 and module.kernel_size[0] < 7:
                    moduleWrap = eval(search_op + 'Op')(module, init_rates, config, self.S)
                    logger.info('Wrap model %s to %s.' % (str(module), str(moduleWrap)))
                    setattr(model, name, moduleWrap)
            elif isinstance(module, BaseOperator):
                pass
            else:
                self.wrap_model(module, config, search_op, init_rates)

    def set_model(self, model, config, search_op='Conv2d', init_rates=None, prefix=""):
        config['model'] = load_config(config['search_log'])
        op = 'nn.' + search_op
        for name, module in model.named_children():
            if prefix == "":
                fullname = name
            else:
                fullname = prefix + '.' + name
            if isinstance(module, eval(op)):
                if module.kernel_size[0] > 1 and module.kernel_size[0] < 7:
                    if isinstance(config["model"][fullname], int):
                        config['model'][fullname] = (config['model'][fullname], )
                    module.dilation = (
                        config['model'][fullname][0],
                        config['model'][fullname][0]
                    )
                    module.padding = (
                        config["model"][fullname][0] * (module.kernel_size[0] - 1) // 2,
                        config["model"][fullname][0] * (module.kernel_size[0] - 1) // 2,
                    )
                    setattr(model, name, module)
                    logger.info('Set module %s dilation as: [%d]' % (fullname, module.dilation[0]))
            elif isinstance(module, BaseOperator):
                pass
            else:
                self.set_model(module, config, search_op, init_rates, fullname)
