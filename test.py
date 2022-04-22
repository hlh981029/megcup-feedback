import os
import time
import numpy as np
import megengine as mge

import yaml
from models import build_model
from datasets import build_dataloader

from criterions import NumpyScores
from logger import create_logger
from utils import get_params_count, AverageMeter, load_pretrained
from searcher.search_engine import Searcher, set_model, wrap_model

from options.parse_opt import parse_option


class Runner:
    def __init__(self, config, logger) -> None:
        self.config = config
        self.logger = logger

        self.dataset_val, self.dataset_test, \
            self.data_loader_val, self.data_loader_test, \
            = build_dataloader(config)

        logger.info(f"Creating model:{config['model']['name']}")

        self.ema_params = None
        self.ema_model = build_model(config)
        self.ema_model.eval()
        if not config['searcher']['finetune']:
            wrap_model(self.ema_model, config['searcher'])
        else:
            set_model(self.ema_model, config['searcher'])
            if config['searcher']['use_warp']:
                wrap_model(self.ema_model, config['searcher'])

        logger.info(str(self.ema_model))
        _param_count = get_params_count(self.ema_model, logger)
        logger.info(f"[Warped Arch] params {_param_count / 1000.0}K")

        self.criterion = NumpyScores()

    def __call__(self):
        config = self.config

        self.ema_params = load_pretrained(self.config, self.logger)

        self.logger.info('Evaluation Started!')
        if config['eval_valid']:
            self.validate()
        else:
            self.test()
        self.logger.info('Evaluation Finished!')
        return

    def validate(self):
        self.ema_model.load_state_dict({k: v.numpy() for k, v in self.ema_params.items()})
        model = self.ema_model
        model.eval()

        batch_time = AverageMeter()
        diff_meter = AverageMeter()

        for idx, (images, targets) in enumerate(self.data_loader_val):
            start = time.time()
            images = mge.tensor(images, dtype='float32')
            targets = mge.tensor(targets, dtype='float32')

            outputs = model(images)

            accs = self.criterion(outputs, targets)

            diff_meter.update(accs['diff'].item(), targets.shape[0])

            batch_time.update(time.time() - start)

            if idx % self.config['print_freq'] == 0:
                self.logger.info(
                    f'Valid: [{idx}/{len(self.data_loader_val)}]\t'
                    f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'diff {diff_meter.val:.5f} ({diff_meter.avg:.5f})')

        score_total = np.log10(100 / diff_meter.avg) * 5
        self.logger.info(
            f'Valid:\t'
            f'diff {float(diff_meter.avg):.4f}\t'
            f'score {score_total:.4f}\t'
            f'time {batch_time.sum:.4f}s')

    def test(self):
        self.ema_model.load_state_dict({k: v.numpy() for k, v in self.ema_params.items()})
        model = self.ema_model
        model.eval()

        batch_time = AverageMeter()
        basename = os.path.basename(self.config['model']['resume'])[:-4]
        fout = open(os.path.join(f'{self.config["output"]}', f'{basename}_result.bin'), 'wb')

        for idx, images in enumerate(self.data_loader_test):
            start = time.time()
            images = mge.tensor(images, dtype='float32')

            outputs = model(images)

            pred = self.dataset_test.reverse_preprocess(outputs.numpy()[:, :, :, :]).clip(0, 65535).astype('uint16')

            fout.write(pred.tobytes())

            batch_time.update(time.time() - start)

            if idx % self.config['print_freq'] == 0:
                self.logger.info(
                    f'Test: [{idx}/{len(self.data_loader_test)}]\t'
                    f'time {batch_time.val:.3f} ({batch_time.avg:.3f})')

        fout.close()


if __name__ == '__main__':
    _, config = parse_option(config_file='options/feedback.yaml')
    seed = config['seed']
    os.makedirs(config['output'], exist_ok=True)

    start_time = time.strftime("%y%m%d-%H%M", time.localtime())
    logger = create_logger(output_dir=config['output'],
                           name=f"{config['tag']}",
                           action=f'eval-{start_time}')

    mge.random.seed(seed)
    np.random.seed(seed)

    path = os.path.join(config['output'], f'eval-{start_time}.yaml')
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Full config saved to {path}")

    logger.info(yaml.dump(config, default_flow_style=False))

    runner = Runner(config, logger)
    runner()
