import yaml
import os
import argparse


def parse_option(config_file=None, eval_mode=False):
    parser = argparse.ArgumentParser('MegCup2022 evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default=config_file, metavar="FILE", help='path to config file', )
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', type=str, metavar='PATH', help='full path is <output>/<model_name>/<tag>')
    parser.add_argument('--eval', action='store_true', default=eval_mode, help='Perform evaluation only')
    parser.add_argument('--submit', action='store_true', help='submit only')

    args, _ = parser.parse_known_args()
    config = parse(args.cfg, args)
    return args, config


def dict2str(value='', key='dict', indent_level=1):
    """dict to string for printing options.
    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.
    Return:
        (str): Option string for printing.
    """
    msg = ' ' * (indent_level * 2) + key + ': (' + type(value).__name__ + ') '
    if isinstance(value, dict):
        msg += '{' + '\n'
        for _key, _value in value.items():
            msg += dict2str(key=_key, value=_value, indent_level=indent_level + 1)
        msg += ' ' * (indent_level * 2) + '}'
    elif isinstance(value, list) or isinstance(value, tuple):
        msg += '[' + '\n'
        for _key, _value in enumerate(value):
            msg += dict2str(key=str(_key), value=_value, indent_level=indent_level + 1)
        msg += ' ' * (indent_level * 2) + ']'
    else:
        msg += str(value)
    if indent_level != 1:
        msg += '\n'
    return msg


def ordered_yaml():
    """Support OrderedDict for yaml.
    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return dict(loader.construct_pairs(node))

    Dumper.add_representer(dict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, args):
    """Parse option file.
    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.
    Returns:
        (dict): Options.
    """
    Loader, _ = ordered_yaml()

    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    assert opt.get('tag') is not None, 'must set tag in option'

    opt.setdefault('output', 'output')
    opt.setdefault('seed', 3407)

    opt['model'].setdefault('resume', '')
    opt['model'].setdefault('pretrain', '')

    opt['data'].setdefault('valid_ref_file_path', '')
    opt['data'].setdefault('valid_gt_file_path', '')

    if args.resume:
        opt['model']['resume'] = args.resume
    if args.output:
        opt['output'] = args.output
    if args.eval:
        opt['eval_valid'] = True
    if args.submit:
        opt['eval_valid'] = False

    opt['output'] = os.path.join(opt['output'], opt['tag'])

    return opt
