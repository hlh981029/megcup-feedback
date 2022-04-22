import os
import numpy as np
from megengine.data import DataLoader
from .competition import Competition


def build_dataloader(config):

    valid_file_path = config['data']['valid_ref_file_path']
    valid_gt_file_path = config['data']['valid_gt_file_path']
    test_file_path = config['data']['test_file_path']

    if config['eval_valid']:
        test_dataset, test_loader = None, None

        assert os.path.exists(valid_file_path)
        assert os.path.exists(valid_gt_file_path)

        content = open(valid_file_path, 'rb').read()
        valid_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
        content = open(valid_gt_file_path, 'rb').read()
        valid_gt = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
        valid_dataset = Competition(valid_ref, valid_gt)
        valid_loader = DataLoader(valid_dataset)

    else:
        valid_dataset, valid_loader = None, None

        assert os.path.exists(test_file_path)
        content = open(test_file_path, 'rb').read()
        samples_test = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
        samples_test = samples_test.astype('float32')

        test_dataset = Competition(samples_test, None)
        test_loader = DataLoader(test_dataset)

    return valid_dataset, test_dataset, valid_loader, test_loader
