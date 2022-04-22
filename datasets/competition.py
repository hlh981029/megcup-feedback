from megengine.data.dataset import Dataset
import numpy as np
from utils import pixel_shuffle, pixel_unshuffle


class Competition(Dataset):
    def __init__(self,
                 bin_input,
                 bin_gt,
                 **kwargs
                 ):

        self.pixel_shuffle = pixel_shuffle
        self.pixel_unshuffle = pixel_unshuffle

        self.bin_input = self.preprocess(bin_input)
        self.bin_gt = self.preprocess(bin_gt)
        self.bin_num = self.bin_input.shape[0]

        if bin_gt is not None:
            assert self.bin_gt.shape[0] == self.bin_input.shape[0], \
                'Binary input and gt size not match!'

    def preprocess(self, x):
        if x is not None:
            x = np.expand_dims(x, 1).astype(np.float32)
            x = self.pixel_unshuffle(x)
            return x / 65535
        return None

    @property
    def demo_ref(self):
        return np.random.rand(1, 4, 128, 128).astype("float32")

    def reverse_preprocess(self, x):
        return self.pixel_shuffle(x * 65535)

    def __getitem__(self, index):
        x = self.bin_input[index]
        if self.bin_gt is not None:
            gt = self.bin_gt[index]
            return x.copy(), gt.copy()
        else:
            return x.copy()

    def __len__(self):
        return self.bin_num
