
from collections import namedtuple
from threading import Thread
from Sim_env.user_equipment import UeRb, UE_RB_ACTION
import numpy as np

SIM_ENV_CONFIG = namedtuple("SIM_ENV_CONFIG", ['n_ue', 'n_episode', 'n_step', 'ue_config', 'action_conversion_f'])

class ENV(Thread):
    def __init__(self, env_name, config):
        Thread.__init__(self)
        self.env_name = env_name
        self.config = config
        self.ue_list = None
        self.true_ue = None
        self.tmp_state = []
        self.n_step = 0
        self.punish_time = 0
        self.overload = 0
        self.active_ue = 0

    phi = np.array([0., 0., 0.25, 0.5, 0.75, 1., 1.])  #前两个也为0应该是考虑到了本身就没包不能乱发的问题
    gamma = 0.9


    def reset(self):  #初始化，并把用户的class装进ue_list
        self.ue_list = []
        for u in range(self.config.n_ue):
            self.ue_list.append(UeRb(u, self.config.ue_config))
        self.tmp_state = []
        state = self.get_state()
        return state

    def get_state(self):  #得到所有用户的state
        self.tmp_state = []
        self.true_ue = []
        self.active_ue = 0
        for u in range(self.config.n_ue): #对在介入范围内的用户接入基站，不在范围内的用户删除其队列
            if self.ue_list[u].channel.connection:
                self.ue_list[u].count += 1
                self.true_ue.append(self.ue_list[u])
                self.tmp_state.append(self.ue_list[u].get_state())  #对每个用户调用ue的get_state()
            else:
                self.ue_list[u].rlc.delete()
                self.ue_list[u].count = 0
            if self.ue_list[u].get_state().q_length:
                self.active_ue += 1

        state_number = []
        for u in range(len(self.true_ue)):
            pct = float(self.tmp_state[u].hol) / float(self.config.ue_config.rlc.d_max)
            state_number.append(pct)

        for u in range(len(self.true_ue)):
            pct = float(self.tmp_state[u].n_rb) / float(self.config.ue_config.channel.total_n_rb) #total_n_rb = 50
            state_number.append(pct)

        state = np.array(state_number, dtype=float)  #得到[k个holpct，k个n_rbpct]
        return state

    def ins_reward(self, action):  #得到所有用户的reward
        rewards = []
        total_rb = 0.0
        for u in range(len(self.true_ue)):
            a_n_rb = self.tmp_state[u].n_rb
            if action[u] == 1 and self.tmp_state[u].q_length:
                total_rb += a_n_rb

        if total_rb > self.config.ue_config.channel.total_n_rb:
            self.overload += 1
            print('overload!!!!!!!!!!!', self.overload)

        for u in range(len(self.true_ue)):
            if action[u] == 1 and self.tmp_state[u].q_length > 0:
                n_rb = round(float(self.tmp_state[u].n_rb) / total_rb * self.config.ue_config.channel.total_n_rb)
                r = self.true_ue[u].step(UE_RB_ACTION(n_rb))     #式14
            else:
                self.true_ue[u].step(UE_RB_ACTION(0))
                r = 0
            rewards.append(r)
        return np.array(rewards, dtype=float)  #接入的ue的reward在这里

    def step(self, action):  #需要返回next_state,reward,done
        action_ = np.copy(action)
        #phi = self.get_phi()
        #rewrite input action here，important！
        print('action', action)
        print('true_ue', len(action))
        action_[action_ >= 0] = 1
        action_[action_ < 0] = 0
        print('action_match', action_)
        r_before = self.ins_reward(action_)
        for u in range(self.config.n_ue):
            self.ue_list[u].channel.change_position()         #这里变成变长以后不能进行rs了

        next_state = self.get_state()
        #phi_next = self.get_phi()
        #shaper = - 1. * (phi - self.gamma * phi_next)
        #rewards = shaper + r_before
        '''
        for u in range(self.config.n_ue):
            
            if self.tmp_state[u].hol < 5 and action_[u] == 1:
                rewards[u] = -5 + rewards[u]
            
            if self.tmp_state[u].hol < 5 and action_[u] == 1 and self.tmp_state[u].q_length != 0:
                self.punish_time += 1

        hol = self.print_hol()
        print('hol', hol[0])
        #print('reward', r_before)
        #print('reward_reshape', rewards)
        '''
        done = 0
        self.n_step += 1
        if self.n_step == self.config.n_episode:
            done = 1
            self.n_step = 0
        return next_state, r_before, r_before, done
    '''
    def step(self, action):  #需要返回next_state,reward,done
        action_ = np.copy(action)
        #rewrite input action here，important！
        action_[action_ > 0] = 1
        action_[action_ <= 0] = 0
        #print('action_match', action_)
        r_before = self.ins_reward(action_)
        rewards = np.zeros(15)
        next_state = self.get_state()
        for u in range(self.config.n_ue):
            if self.tmp_state[u].hol < 5 and action_[u] == 1 and self.tmp_state[u].q_length != 0:
                #rewards[u] = -5 + r_before[u]
                rewards[u] = r_before[u]
                self.punish_time += 1
            else:
                rewards[u] = r_before[u]
        hol = self.print_hol()
        print('hol', hol[0])
        print('reward', r_before)
        print('reward_reshape', rewards)

        done = 0
        self.n_step += 1
        if self.n_step == self.config.n_step:
            done = 1
            self.n_step = 0
        return next_state, rewards, r_before, done, hol
    '''
    def print_hol(self):
        hol = []
        for u in range(self.config.n_ue):
            hol.append(self.tmp_state[u].hol)
        return hol

    def get_phi(self):
        phi_list = []
        for u in range(len(self.true_ue)):
            phi_list.append(self.phi[self.tmp_state[u].hol])
        return np.array(phi_list, dtype=float)

    def getpunish_time(self):
        return self.punish_time

    def getoverload_time(self):
        return self.overload