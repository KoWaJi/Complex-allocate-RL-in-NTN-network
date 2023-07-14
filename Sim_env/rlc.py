from collections import deque
from collections import namedtuple
from Sim_env.math_models import *
from Sim_env.reward_functions import *
import numpy as np

RLC_CONFIG = namedtuple("RLC_CONFIG", ['packet_size', 'max_size', 'packet_p', 'd_min', 'd_max', 'hol_reward_f'])
RLC_STATE = namedtuple("RLC_STATE", ['n_packet', 'hol', 'n_byte'])
RLC_ACTION = namedtuple("RLC_ACTION", ['packet'])


class Rlc:
    def __init__(self, id, config):
        self.id = id
        self.config = config
        self.packet = namedtuple("Packet", ['time', 'packet_size'])
        self.queue = deque(maxlen=config.max_size)
        self.time_step = 0
        self.n_discard = 0        #该用户丢弃包的总数量
        self.sum_packet = 0       #该用户到达包的总数量（不随用户离开而重置）

    def get_state(self):
        return RLC_STATE(n_packet=len(self.queue),
                         hol=self.get_hol(),
                         n_byte=self.get_n_byte_total())

    def get_hol(self):  #排队时延
        if self.queue:
            return self.time_step - self.queue[0].time
        else:
            return 0

    def get_n_byte_total(self):
        ret = 0
        for x in range(len(self.queue)):
            ret += self.queue[x].packet_size

        return ret

    def delete(self):  #初始化队列
        self.queue.clear()

    def step(self, action):  #这里应该就是action对state的影响了
        ret = 0.
        for i in range(action.packet):
            self.pop()
            ret += self.get_hol_reward()
        self.time_step += 1
        discard_by_outtime = self.discard()
        self.n_discard += discard_by_outtime #超时被丢弃的包数量
        ret -= discard_by_outtime
        return ret

    def get_hol_reward(self):  #在抖动时间内就返回1
        if self.config.hol_reward_f is None:
            if self.queue and self.get_hol() <= self.config.d_max and self.get_hol() >= self.config.d_min:
                return 1.
            else:
                return 0.
        else:
            return self.config.hol_reward_f(self.get_hol(), self.config.d_min, self.config.d_max)  #对于rlc，在时延内
                                                                                   #调度或者不进行调度return为0，时延前调度或者
                                                                                    #包超时丢弃return为-1

    def pop(self):  #pop的输入是资源块数量
        if self.queue:

            self.queue.popleft()

    def push(self):
        """
        :return: the number of packet is discard
        """
        packet = np.random.poisson(self.config.packet_p)  #数据包的到达符合泊松分布
        if packet != 0:
            record = packet
            for i in range(packet):
                if len(self.queue) < self.queue.maxlen:
                    self.queue.append(self.packet(self.time_step, self.config.packet_size))  # packet_size是32
                    self.sum_packet += 1
                else:
                    record = i
                    break
            return packet - record
        else:
            return 0


    def discard(self):   #时延过大了就丢弃了，返回丢弃的数量
        n_discard = 0
        while self.queue and self.get_hol() > self.config.d_max:
            self.queue.popleft()
            n_discard += 1
            print('discard!!')
        return n_discard

if __name__ == '__main__':
    config = RLC_CONFIG(packet_size=32,
                                     max_size=100,
                                     packet_p=0.1,
                                     d_min=5,
                                     d_max=6,
                                     hol_reward_f=hol_flat_reward)
    A = Rlc(1, config)
    action = RLC_ACTION(['1'])
    for i in range(30):
        A.step(action)
        b = A.get_state()
        print(b)
















