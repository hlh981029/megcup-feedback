import megengine as mge
import megengine.functional as F
import os
import tabulate
import numpy as np


def pixel_unshuffle(x, block=2):
    assert len(x.shape) == 4
    n, c, h, w = x.shape
    x = x.reshape((n, c, h // block, block, w // block, block)) \
         .transpose((0, 1, 3, 5, 2, 4)) \
         .reshape((n, c * block * block, h // block, w // block))
    return x


def pixel_shuffle(x, block=2):
    assert len(x.shape) == 4
    n, c, h, w = x.shape
    x = x.reshape((n, c // (block * block), block, block, h, w)) \
         .transpose((0, 1, 4, 2, 5, 3)) \
         .reshape((n, c // (block * block), h * block, w * block))
    return x


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_pretrained(config, logger, config_key='resume', ema_key='ema_model'):

    logger.info(f"====> Load checkpoint form {config['model'][config_key]}......")
    assert os.path.exists(config['model'][config_key]), \
        f"file not found {config['model'][config_key]}"
    checkpoint = mge.load(config['model'][config_key], map_location=lambda dev: 'cpu0')

    return checkpoint[ema_key]


def load_ema_param(config, logger, config_key='resume', ema_key='ema_model'):
    logger.info(f"====> Load ema params form {config['model'][config_key]}......")
    checkpoint = mge.load(config['model'][config_key], map_location=lambda dev: 'cpu0')
    return checkpoint[ema_key]


def dict2table(list_of_dict, header):
    table_data = [header]
    for d in list_of_dict:
        row = []
        for h in header:
            v = ""
            if h in d:
                v = d[h]
            row.append(v)
        table_data.append(row)
    return table_data


def get_params_count(model, logger):
    header = [
        'name', 'dim', 'size', 'percent'
    ]
    dicts = []
    _param_count = sum(p.size for p in model.parameters())
    for n, p in model.named_parameters():
        dic = {}
        dic['name'] = n
        dic['dim'] = p.shape
        dic['size'] = p.size
        dic['percent'] = f"{p.size/_param_count*100.0:.2f}%"
        dicts.append(dic)

    logger.info(
        "param stats: \n" + tabulate.tabulate(dict2table(dicts, header=header))
    )
    return _param_count


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


# Registry
# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501


class Registry():
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (name not in self._obj_map), (f"An object named '{name}' was already registered "
                                             f"in '{self._name}' registry!")
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


def convert_deprecated_norm(norm_name):
    convert_dict = {
        'NewLayerNorm': 'ChannelLayerNorm',
        'LayerNorm': 'InstanceNorm2d',
        'InstanceNorm': 'InstanceNorm3d',
        'HalfInstanceLayerNorm': 'HalfInstanceNorm2d',
        'HalfMyInstanceNorm': 'HalfInstanceNorm3d',
    }
    print(norm_name)
    if norm_name in convert_dict.keys():
        print('****************************** Noted *********************************')
        print(f': "{norm_name}" is deprecated! Please use "{convert_dict[norm_name]}"')
        print('**********************************************************************')

        norm_name = convert_dict[norm_name]
    return norm_name


ARCH_REGISTRY = Registry('arch')
