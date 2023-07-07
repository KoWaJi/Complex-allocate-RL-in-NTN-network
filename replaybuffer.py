import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, args):
        self.exp = {'s': [], 'a': [],
               'a_logprob': [], 'r': [],
               's_': [], 'dw': [],
               'done': []}

        self.count = 0
        self.device = torch.device('cuda:2')

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.exp['s'].append(s)
        self.exp['a'].append(a)
        self.exp['a_logprob'].append(a_logprob)
        self.exp['r'].append(r)
        self.exp['s_'].append(s_)
        self.exp['dw'].append(dw)
        self.exp['done'].append(done)
        self.count += 1

    def returnzero(self):
        self.exp = {'s': [], 'a': [],
               'a_logprob': [], 'r': [],
               's_': [], 'dw': [],
               'done': []}
        self.count = 0

    def numpy_to_tensor(self):
        s = self.exp['s']   #长度不同不能转为tensor
        a = self.exp['a']
        a_logprob = self.exp['a_logprob']
        r = self.exp['r']
        s_ = self.exp['s_']
        dw = torch.tensor(self.exp['dw']).to(self.device)
        done = torch.tensor(self.exp['done'])
        '''
        s = np.array(self.exp['s'])
        s = s.astype(float)
        s = torch.tensor(s).to(self.device)
        #s = torch.tensor(self.exp['s'], dtype=torch.float).to(self.device)
        a = torch.tensor(self.exp['a'], dtype=torch.float).to(self.device)
        a_logprob = torch.tensor(self.exp['a_logprob'], dtype=torch.float).to(self.device)
        r = torch.tensor(self.exp['r'], dtype=torch.float).to(self.device)
        s_ = torch.tensor(self.exp['s_'], dtype=torch.float).to(self.device)
        dw = torch.tensor(self.exp['dw'], dtype=torch.float).to(self.device)
        done = torch.tensor(self.exp['done'], dtype=torch.float).to(self.device)
        '''
        '''
        experiences = random.sample(self.memory, k=self.args.mini_batch_size)
        s = torch.tensor(np.vstack([e.s for e in experiences if e is not None]), dtype=torch.float).cuda()  #np.vstack输入tuple返回array
        a = torch.tensor(np.vstack([e.a for e in experiences if e is not None]), dtype=torch.float).cuda()
        a_logprob = torch.tensor(np.vstack([e.a_logprob for e in experiences if e is not None]), dtype=torch.float).cuda()
        r = torch.tensor(np.vstack([e.r for e in experiences if e is not None]), dtype=torch.float).cuda()
        s_ = torch.tensor(np.vstack([e.s_ for e in experiences if e is not None]), dtype=torch.float).cuda()
        done = torch.tensor(np.vstack([e.done for e in experiences if e is not None]), dtype=torch.float).cuda()
        dw = torch.tensor(np.vstack([e.dw for e in experiences if e is not None]), dtype=torch.float).cuda()
        '''
        return s, a, a_logprob, r, s_, dw, done