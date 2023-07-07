from collections import deque
from collections import namedtuple
from Sim_env.math_models import *
from Sim_env.reward_functions import *

RLC_CONFIG = namedtuple("RLC_CONFIG", ['packet_size', 'max_size', 'packet_p', 'd_min', 'd_max', 'hol_reward_f'])
RLC_STATE = namedtuple("RLC_STATE", ['n_packet', 'hol', 'n_byte'])
RLC_BINARY_TX_ACTION = namedtuple("RLC_BINARY_TX_ACTION", ['tx'])


class Rlc:
    def __init__(self, id, config):
        self.id = id
        self.config = config
        self.packet = namedtuple("Packet", ['time', 'packet_size'])
        self.queue = deque(maxlen=config.max_size)
        self.time_step = 0
        self.n_packet = 0

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
        self.queue = deque(maxlen=self.config.max_size)

    def step(self, action):  #这里应该就是action对state的影响了
        ret = 0.
        if action.tx: #action.tx = action.n_rb
            ret = self.get_hol_reward()
            self.pop()
        n_discard = self.push() #满了才会返回1
        self.time_step += 1
        n_discard += self.discard() #当然被丢弃了就不会返回1了
        '''
        n_packet = self.n_packet
        n0 = n_discard + n_packet
        if n0 == 0:
            packetloss = 0
        else:
            packetloss = n_discard / (n_discard + n_packet)
        '''
        return ret
    def get_hol_reward(self):  #在抖动时间内就返回1
        if self.config.hol_reward_f is None:
            if self.queue and self.get_hol() <= self.config.d_max and self.get_hol() >= self.config.d_min:
                return 1.
            else:
                return 0.
        else:
            if self.queue:
                return self.config.hol_reward_f(self.get_hol(), self.config.d_min, self.config.d_max)
            else:
                return 0.

    def pop(self):
        if self.queue:
            self.queue.popleft()

    def push(self):
        """
        :return: the number of packet is discard
        """
        if p_true(self.config.packet_p):  #10%
            self.n_packet += 1        
            if len(self.queue) == self.queue.maxlen:
                return 1
            else:
                self.queue.append(self.packet(self.time_step, self.config.packet_size))  #packet_size是32
                return 0
        k = self.queue
        return 0

    def discard(self):   #时延过大了就丢弃了，返回丢弃的数量
        n_discard = 0
        while self.queue and self.get_hol() > self.config.d_max:
            self.queue.popleft()
            n_discard += 1
        return n_discard

if __name__ == '__main__':
    config = RLC_CONFIG(packet_size=32,
                                     max_size=100,
                                     packet_p=0.1,
                                     d_min=5,
                                     d_max=6,
                                     hol_reward_f=hol_flat_reward)
    A = Rlc(1, config)
    action = RLC_BINARY_TX_ACTION(['1'])
    for i in range(30):
        A.step(action)
        b = A.get_state()
        print(b)
















