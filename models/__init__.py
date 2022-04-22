import importlib
from copy import deepcopy
from os import path as osp

from utils import scandir, ARCH_REGISTRY

from .blocks.decorator import NormDecorator

__all__ = ['build_model']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'models.{file_name}') for file_name in arch_filenames]


def build_model(config):
    config = deepcopy(config)
    model_type = config['model']['name']
    net = ARCH_REGISTRY.get(model_type)(**config['model'].get('args', {}))
    decorator = get_decorator(config['data'].get('datanorm'))
    if decorator is not None:
        from megengine import module as M

        class DecoratedNet(M.Module):
            def __init__(self):
                super().__init__(net.name)
                self.net = net

            def forward(self, x):
                x = decorator.norm(x)
                x = self.net(x)
                if isinstance(x, (list, tuple)):
                    return [decorator.renorm(xx) for xx in x]
                return decorator.renorm(x)
        return DecoratedNet()
    return net


def get_decorator(datanorm_type):
    """
    available keys:
        full_dn         : mean std based on full dataset, [0, 1]
        dn              : mean std based on dataset, [0, 1]
    """
    try:
        return NormDecorator(datanorm_type)
    except KeyError:
        return None
