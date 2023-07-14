from collections import namedtuple
from Sim_env.channel import Channel, CHANNEL_UNKNOWN_ERROR_ACTION
from Sim_env.math_models import *
from Sim_env.rlc import Rlc, RLC_ACTION

UE_CONFIG = namedtuple("UE_CONFIG", ['channel', 'rlc', 'error_rate'])
#UE_RB_MODEL_FREE_STATE = namedtuple("UE_RB_STATE", ['snr_db', 'hol', 'packets'])
UE_RB_ACTION = namedtuple("UE_RB_ACTION", ['n_rb'])
UE_RB_STATE = namedtuple("UE_RB_STATE", ['snr_db', 'hol', 'packets'])
CHANNEL_STATE = namedtuple("CHANNEL_STATE", ['snr_db'])

class UeRb:
    """
    the UE object:
        Action : the number of rb allocated to this UE
        State: the channel and rlc queue state
    """
    def __init__(self, id, config):
        self.id = id
        self.config = config
        self.channel = Channel(id, config.channel)
        self.rlc = Rlc(id, config.rlc)
        #self.count = 0      #该计数器用来统计用户接入基站后经过了多少ttl，接入基站时发送一次CSI，然后每经过五个ttl再发送一次CSI，对用户单
                            #独进行计数
        #self.channel_state = CHANNEL_STATE(0)  #用来存得不到CSI时的旧CSI，每五个ttl更新一次

    def step(self, action):  #在这里完成action和env的交互

        channel_action = CHANNEL_UNKNOWN_ERROR_ACTION(action.n_rb, self.config.rlc.packet_size)
        channel_reward = self.channel.step(channel_action)  #channel.step
        packet_size = self.config.rlc.packet_size   #在这里算出分配的资源块最多能在保证误码率的条件下对应几个数据包
        packet = 0
        for x in range(len(self.rlc.queue), -1, -1):
            e = tx_error_rate_for_n_bytes(packet_size * x,
                                          action.n_rb,
                                          db_to_dec(self.channel.get_snr_db()),
                                          self.config.channel.T_f,
                                          self.config.channel.rb_bw)

            if e <= self.config.error_rate:
                packet = x
                break

        rlc_action = RLC_ACTION(packet)
        hol_reward = self.rlc.step(rlc_action) #rlc.step
        return channel_reward * hol_reward, packet

    def get_state(self):
        channel_state = self.channel.get_state()
        rlc_state = self.rlc.get_state()
        '''
        packet_size = self.config.rlc.packet_size
        for x in range(MAX_NUM_RB):  # MAX_NUM_RB = 50 from math_models
            n_rb = x + 1
            e = tx_error_rate_for_n_bytes(packet_size,
                                              n_rb,
                                              db_to_dec(self.channel_state.snr_db),
                                              self.config.channel.T_f,
                                              self.config.channel.rb_bw)

            if e <= self.config.error_rate:
                break
        '''
        return UE_RB_STATE(channel_state.snr_db, rlc_state.hol, rlc_state.n_packet)
