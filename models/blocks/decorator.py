import numpy as np
import megengine as mge


class NormDecorator():
    def __init__(self, norm_type=''):
        type2func = {
            'dn': lambda: self.dataset_norm(),
            'full_dn': lambda: self.full_dataset_norm(),
        }
        self.norm, self.renorm = type2func[norm_type]()

    def dataset_norm(self):
        mean = mge.tensor(np.array([0.08976055, 0.1122945, 0.11342434, 0.09182885]).reshape((1, 4, 1, 1)), dtype='float32')
        std = mge.tensor(np.array([0.06073199, 0.09204482, 0.09400411, 0.06862978]).reshape((1, 4, 1, 1)), dtype='float32')
        self.mean = mge.Parameter(mean, is_const=True)
        self.std = mge.Parameter(std, is_const=True)

        def norm(x):
            return (x - self.mean) / self.std

        def renorm(x):
            return x * self.std + self.mean

        return norm, renorm

    def full_dataset_norm(self):
        mean = mge.tensor(np.array([0.08958147, 0.11184454, 0.11297596, 0.09162903]).reshape((1, 4, 1, 1)), dtype='float32')
        std = mge.tensor(np.array([0.06037381, 0.09139108, 0.09338845, 0.06839059]).reshape((1, 4, 1, 1)), dtype='float32')
        self.mean = mge.Parameter(mean, is_const=True)
        self.std = mge.Parameter(std, is_const=True)

        def norm(x):
            return (x - self.mean) / self.std

        def renorm(x):
            return x * self.std + self.mean

        return norm, renorm

    def __call__(self, forward):
        def dec_forward(sf, x):
            x = self.norm(x)
            y = forward(sf, x)
            return self.renorm(y)
        return dec_forward
