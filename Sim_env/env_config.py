from Sim_env.channel import CHANNEL_CONFIG
from Sim_env.reward_functions import *
from Sim_env.rlc import RLC_CONFIG
from Sim_env.env import SIM_ENV_CONFIG
from Sim_env.user_equipment import UE_CONFIG


class env_config_helper():
    def __init__(self):
        self.N_UE = 120
        self.N_EPISODE = 500
        self.N_STEP = 500

        self.ERROR_RATE = 1e-5  
        self.PACKET_SIZE = 32
        self.p = 0.4
        self.D_MIN = 5
        self.D_MAX = 6

        self.TOTAL_N_RB = 50

        self.T_f = 1.25e-4   
        self.rb_bw = 180e3   

        assert self.D_MIN <= self.D_MAX

        self.D_MIN_to_D_MAX_pct = float(self.D_MIN) / float(self.D_MAX)

    def reload_config(self):
        self.rlc_config = RLC_CONFIG(packet_size=self.PACKET_SIZE,
                                     max_size=100,   
                                     packet_p=self.p,
                                     d_min=self.D_MIN,
                                     d_max=self.D_MAX,
                                     hol_reward_f=hol_flat_reward)  

        
        self.channel_config = CHANNEL_CONFIG(max_dis=200,
                                             step_dis=5,
                                             move_p=0.1,
                                             tx_power=20,
                                             noise_power=-90,
                                             T_f=self.T_f,
                                             rb_bw=self.rb_bw,
                                             total_n_rb=self.TOTAL_N_RB)

        self.ue_config = UE_CONFIG(channel=self.channel_config,   
                                   rlc=self.rlc_config,
                                   error_rate=self.ERROR_RATE)

        self.sim_env_config = SIM_ENV_CONFIG(n_ue=self.N_UE,     
                                             n_episode=self.N_EPISODE,
                                             n_step=self.N_STEP,
                                             ue_config=self.ue_config,
                                             action_conversion_f=None)