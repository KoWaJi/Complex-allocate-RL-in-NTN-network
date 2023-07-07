
import random
from collections import namedtuple
from scipy.stats import rice
from Sim_env.math_models import *


CHANNEL_CONFIG = namedtuple("CHANNEL_CONFIG",
                            ['max_dis', 'step_dis', 'move_p', 'tx_power', 'noise_power', 'T_f', 'rb_bw', 'total_n_rb'])
CHANNEL_STATE = namedtuple("CHANNEL_STATE", ['snr_db'])
CHANNEL_UNKNOWN_ERROR_ACTION = namedtuple("CHANNEL_UNKNOWN_ERROR_ACTION", ['n_rb', 'n_byte'])

class Channel:
    def __init__(self, id, config):  
        self.id = id
        self.config = config
        self.dis = 0
        self.connection = True  #connection用来判断用户是否接入基站
        self.init_distance()

        self.scale = 0.559  
        self.shape = 0.612 / self.scale  
        self.small_scale_gain = rice.rvs(self.shape, scale=self.scale)

    def get_state(self):
        return CHANNEL_STATE(self.get_snr_db())

    def get_snr_db(self) -> float:

        snr = distance_to_snr(self.dis, self.config.tx_power, self.config.noise_power)

        snr += dec_to_db(self.small_scale_gain)

        if snr > 20.:
            return 20.
        else:
            return snr

    def step(self, action):  #返回式17的奖励，并依概率变动用户位置
        err = 0.
        if action.n_rb > 0:
            err = tx_error_rate_for_n_bytes(action.n_byte, action.n_rb, db_to_dec(self.get_snr_db()),
                                                self.config.T_f,
                                                self.config.rb_bw)  # 式11，式12

            if action.n_rb >= self.config.total_n_rb and err < 1e-5: #分配了过量的资源块
                err = 1e-5

            if err <= 1e-5:  #误码率能满足要求
                ret = 5.
            else:
                ret = - math.log10(err)  # overload的情况误码率不能满足要求，获得的reward是不到5的
        else:
            ret = 0.

        #self.change_position()

        return ret

    def change_position(self):  #1e-4的概率变动位置，有0.2的概率改变信道状态
        if p_true(self.config.move_p):
            if p_true(0.5):
                self.increase_distance()
                if self.dis > self.config.max_dis / 2:
                    self.connection = False
            else:
                self.decrease_distance()
                if self.dis <= self.config.max_dis / 2:
                    self.connection = True
        
        if p_true(0.2):
            self.small_scale_gain = rice.rvs(self.shape, scale=self.scale)  


    def init_distance(self):
        initial_steps = random.randint(0, int(self.config.max_dis / self.config.step_dis))  
        for x in range(initial_steps):
            self.increase_distance()
        if self.dis > self.config.max_dis / 2:
            self.connection = False

    def increase_distance(self):
        if self.dis + self.config.step_dis <= self.config.max_dis:
            self.dis += self.config.step_dis

    def decrease_distance(self):
        if self.dis - self.config.step_dis >= 0:
            self.dis -= self.config.step_dis



'''
    A channel object
        Action: the number of RB, the number of bytes
        Reward: 1 - tx error rate
'''
