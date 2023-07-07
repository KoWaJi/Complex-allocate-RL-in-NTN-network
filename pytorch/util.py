from datetime import datetime  # 日期时间对象，常用的属性有year、month、day
from time import time          # 时间对象

import numpy as np
import torch

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, requires_grad=False, dtype=FLOAT):     # 不需要保存梯度可以定义为False，为了优化内存使用，默认的tensor是不需要梯度的
    t = torch.from_numpy(ndarray)    # 从numpy.ndarray创建一个张量，返回的张量和ndarray共享一个内存，张量的修改会反映在ndarry中
    t.requires_grad_(requires_grad)  # 是通用数据结构Tensor的一个属性，用于说明当前量是否需要在计算中保留对应的梯度信息
    if USE_CUDA:
        return t.type(dtype).to(torch.cuda.current_device())  # 将模型加载到to()内的设备中
    else:
        return t.type(dtype)


def soft_update_inplace(target, source, tau):  # tau=2🥧
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update_inplace(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def add_param_noise_inplace(target, std=0.01):
    for target_param in list(target.parameters()):
        d = np.random.randn(1)
        d = d * std
        d = to_tensor(d, requires_grad=False)
        target_param.data.add_(d)


def get_current_time_str():
    return datetime.now().strftime("%Y-%B-%d-%H-%M-%S")


def counted(f):  # 与@之后的函数名一致，也就是counted
    def wrapped(self, *args):  # wrapped是装饰器
        self.n_step += 1
        return f(self, *args)

    return wrapped


def timed(f):
    def wrapped(self, *args):
        ts = time()
        result = f(self, *args)
        te = time()
        print('%s func:%r took: %2.4f sec' % (self, f.__name__, te - ts))
        return result

    return wrapped


if __name__ == '__main__':
    print(USE_CUDA)
    # print(torch.normal(2, 3, size=(1, 1)))
    print(np.random.randn(1))
