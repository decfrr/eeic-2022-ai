import gym
from gym import spaces
import numpy as np
from functools import reduce


class Utils:
    """こまごました関数群"""

    @classmethod
    def get_flatten_function_and_size(cls, space):
        """観測をflatにする関数と、flatにしたときのサイズを返す

        観測空間が与えられると、その空間の要素をflatにする関数を返し、
        ついでにflatにしたあとのサイズも返す
        flatにできない (flatten処理が未実装な) 空間が与えられた場合はエラーを吐く

        Args:
            space (gym.Space): flatにしたい観測空間

        Returns:
            flatten (function gym.Spaceの要素 -> numpy.array)
                : 要素をflatな (1次元の) numpy.arrayにする関数
            size (int): flatになった要素のサイズ

        """

        if isinstance(space, spaces.Discrete):
            # one-hot expression
            size = space.n

            def flatten(x):
                return [1 if x == i else 0 for i in range(size)]

        elif isinstance(space, spaces.Box):
            shape = space.shape
            if len(shape) != 1:
                raise NotImplementedError()
            size = shape[0]

            def flatten(x):
                return x

        elif isinstance(space, spaces.Dict):
            k_f_s = ([[k, cls.get_flatten_function_and_size(v)] for k, v in space.spaces.items()])
            size = sum([s for k, (f, s) in k_f_s])

            def flatten(x):
                ret = []
                for k, (f, s) in k_f_s:
                    ret += list(f(x[k]))
                return ret
        else:
            raise NotImplementedError()

        return (lambda x: np.array(flatten(x), dtype=np.float32), size)
