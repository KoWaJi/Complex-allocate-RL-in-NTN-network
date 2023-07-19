import numpy
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal
from tb_logger import GLOBAL_LOGGER
from util import *
from torch_geometric.nn import GCNConv
from GCN import GCN
from torch.distributions import Categorical


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)

'''
class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()

        self.n_states = args.state_dim
        self.n_hidden_filters = args.hidden_width
        self.n_actions = args.action_dim
        self.hidden1 = nn.Linear(in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

        self.critic = nn.Sequential(
            self.hidden1,
            nn.ReLU(),
            self.hidden2,
            nn.ReLU(),
            self.q_value,
        )
'''
'''
class Agent(nn.Module):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.max_action = args.max_action

        self.hidden1 = nn.Linear(args.state_dim, args.hidden_width)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(args.hidden_width, args.hidden_width)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.mu = nn.Linear(args.hidden_width, args.action_dim)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(args.hidden_width, args.action_dim)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

        self.critic = Critic(args)


    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = log_std.clamp(min=-5, max=2).exp()  # [-20, 2]
        return mu, log_std

    def get_value(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.critic.critic(x)


    def get_dist(self, s):
        mu, log_std = self.forward(s)
        std = log_std.exp()
        #std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mu, std)  # Get the Gaussian distribution
        return dist

'''

class Agent(nn.Module):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.GCN1 = GCN(3, args.embedding_width1, args.embedding_width2)   #可扩展性  #这里最好将gcn输出维度这个超参数变量化放进args
        self.fc1 = nn.Linear(2 * args.embedding_width2, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)

        self.fc4 = nn.Linear(2 * args.embedding_width2, args.hidden_width)
        self.fc5 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc6 = nn.Linear(args.hidden_width, 1)
        self.device = torch.device('cuda:2')

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)
            orthogonal_init(self.fc5)
            orthogonal_init(self.fc6)

        self.gcn = nn.Sequential(
            self.GCN1,
        )
        self.critic = nn.Sequential(
            self.fc1,
            nn.LeakyReLU(),
            self.fc2,
            nn.LeakyReLU(),
            self.fc3,
        )
        self.actor = nn.Sequential(
            self.fc4,
            nn.LeakyReLU(),
            self.fc5,
            nn.LeakyReLU(),
            self.fc6,
            nn.Softmax(),
        )

        self.log_std = nn.Parameter(torch.zeros(1), requires_grad=True)
    '''
    def gcn_output(self, x, edge_index):
        x = x.type(torch.float32)  #不知道为什么x变成了double（float64），所以要换成pytorch默认的float32精度计算
        x = self.GCN1(x, edge_index)
        x = F.relu(x)
        x = self.GCN2(x, edge_index)
        return x
    '''

    def get_value(self, x, edge_index):   #critic的输出最后取了求和平均
        #x = self.gcn_output(x, edge_index)
        x = self.gcn((x, edge_index))
        x = self.critic(x)
        average = sum(x)    #? 应不应该做这个平均呢
        return average

    def get_dist(self, x, edge_index):
        #x = self.gcn_output(x, edge_index)
        x = self.gcn((x, edge_index))
        mean = self.actor(x) #用高斯分布采样代替softmax
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


    def get_action(self, x, edge_index):
        x = self.gcn((x, edge_index))
        x = self.actor(x)
        return x


class PPO_continuous():
    def __init__(self, args):
        self.policy_dist = args.policy_dist
        #self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr  # Learning rate of agent
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.device = torch.device('cuda:2')  # 使用cuda，device=cuda

        self.agent = Agent(args)
        self.agent = self.agent.to(self.device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)
            #print(self.optimizer.param_groups)
        else:
            self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.lr)


    def evaluate(self, x, edge_index):  # When evaluating the policy, we select the action with the highest probability
        a = self.agent.get_action(x, edge_index)
        #a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
        return a.detach().cpu().numpy().flatten()


    def choose_action(self, x, edge_index):
        with torch.no_grad():
            dist = self.agent.get_dist(x, edge_index)
            a = dist.sample()  # Sample the action according to the probability distribution
            #a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
            #print('action', a)
            a_logprob = dist.log_prob(a)  # The log probability density of the action
        a = a.cpu()
        a_logprob = a_logprob.cpu()
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    '''
    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        s = s.to(self.device)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.agent.get_dist(s)
                a = dist.rsample()  # Sample the action according to the probability distribution
                a_t = torch.tanh(a)
                action = a_t * self.action_scale + self.action_bias
                a_logprob = dist.log_prob(value=a)  # The log probability density of the action
                a_logprob = a_logprob - torch.log(self.action_scale * (1 - a_t ** 2) + 1e-6)
        else:
            with torch.no_grad():
                dist = self.agent.get_dist(s)
                a = dist.rsample()  # Sample the action according to the probability distribution
                a_t = torch.tanh(a)
                action = a_t * self.action_scale + self.action_bias
                a_logprob = dist.log_prob(value=a)  # The log probability density of the action
                a_logprob = a_logprob - torch.log(self.action_scale * (1 - a_t ** 2) + 1e-6)
        action = action.cpu()
        a_logprob = a_logprob.cpu()
        return action.numpy().flatten(), a_logprob.numpy().flatten()
    '''

    def create_x(self, s):
        s = list(s)
        true_ue = int(len(s) / 3)
        insert_gNB = [0]
        s = insert_gNB + s[:true_ue] + insert_gNB + s[true_ue: 2 * true_ue] + insert_gNB + s[2 * true_ue:]
        true_ue += 1
        snr = torch.tensor(s[:true_ue]).reshape(true_ue, 1)
        hol = torch.tensor(s[true_ue: 2 * true_ue]).reshape(true_ue, 1)
        packet = torch.tensor(s[2 * true_ue:]).reshape(true_ue, 1)
        x = torch.cat((snr, hol, packet), dim=1)
        return x.float().to(self.device)

    def create_edge_index(self, s):
        true_ue = int(len(s) / 3)
        d = [true_ue]
        for i in range(true_ue):
            d.append(1)
        D = np.diag(d)
        A = np.zeros((true_ue + 1, true_ue + 1))
        for i in range(true_ue):
            A[0][i + 1] = 1
            A[i + 1][0] = 1
        edge_index = D + A
        return torch.tensor(edge_index).float().to(self.device)

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        #adv = []
        advnorm = []
        #adv = torch.tensor(adv)
        advnorm = torch.tensor(advnorm)
        deltas = []
        x_batch = []
        edge_index_batch = []
        vs = []
        with torch.no_grad():  # adv and v_target have no gradient
            for i in range(self.batch_size):
                x = self.create_x(s[i])
                x_batch.append(x)
                edge_index = self.create_edge_index(s[i])
                edge_index_batch.append(edge_index)
                x_ = self.create_x(s_[i])
                edge_index_ = self.create_edge_index(s_[i])
                single_vs = self.agent.get_value(x, edge_index)
                vs_ = self.agent.get_value(x_, edge_index_)
                vs.append(single_vs)
                reward = sum(r[i])/len(r[i])        #由于reward也是不固定的，这里还是取了平均值来计算
                reward = torch.tensor(reward).to(self.device)
                d = reward + self.gamma * (1.0 - dw[i]) * vs_ - single_vs   #gae的delta计算       #奇怪这里dw报了bool不能相减的错，那之前ppo里面为什么没报错啊
                deltas.append(d)
            vs = torch.tensor(vs).view(-1, 1).to(self.device)
            deltas = torch.tensor(deltas).view(-1, 1)
            deltas = deltas.cpu()
            done = done.cpu()

            for k in zip(*deltas):  #竖着来看deltas,本意是想对每个用户单独进行计算
                gae = 0
                advantage = []
                k = numpy.asarray(k)
                for delta, d in zip(reversed(k), reversed(done.flatten().numpy())):
                    gae = delta + self.gamma * self.lamda * gae * (1.0 - d)  #对每步计算其advantage
                    advantage.insert(0, gae)
                advantage = torch.tensor(advantage, dtype=torch.float).view(-1, 1)
                #拼接成2048*15的adv矩阵
                #adv = torch.cat((adv, advantage), dim=1)

            for i in zip(*advantage):   # Trick 1:advantage normalization
                i = torch.tensor(i, dtype=torch.float)
                i = ((i - i.mean()) / (i.std()))
                i = i.view(-1, 1)
                advnorm = i
            advnorm = advnorm.to(self.device)

            #adv = torch.tensor(adv)
            advantage = advantage.to(self.device)           #前面advnorm化了但在算Vt的时候并没有哦
            v_target = advantage + vs

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                print('learn')
                ratios = torch.tensor([]).to(self.device)
                v_s = torch.tensor([]).to(self.device)
                dist_entropy = torch.tensor([]).to(self.device)
                for i in index:
                    single_v_s = self.agent.get_value(x_batch[i], edge_index_batch[i])
                    v_s = torch.cat([v_s, single_v_s])
                    dist_now = self.agent.get_dist(x_batch[i], edge_index_batch[i])
                    single_dist_entropy = dist_now.entropy().sum(0, keepdim=True) / len(dist_now.entropy()) #shape(1 X 1),这里也取了平均不知道对不对
                    dist_entropy = torch.cat([dist_entropy, single_dist_entropy])  #主要用来衡量收敛情况
                    a_ = torch.tensor(a[i]).view(-1, 1).to(self.device)
                    a_logprob_now = dist_now.log_prob(a_)
                    # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                    logratios = a_logprob_now.sum(0, keepdim=True) / len(a_logprob_now) - torch.tensor(a_logprob[i]).to(self.device).view(-1, 1).sum(0, keepdim=True) / len(a_logprob[i]) #两个维度有可能是不一样的所以取平均，然后一个列一个行
                    singleratios = logratios.exp()

                    ratios = torch.cat([ratios, singleratios])
                ratios = ratios.view(-1, 1)
                v_s = v_s.view(-1, 1)



                '''
                #use target-kl threhold
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratios - 1) - logratios).mean()
                '''
                if self.use_adv_norm:
                    surr1 = ratios * advnorm[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advnorm[index]
                    #actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                else:
                    surr1 = ratios * advantage[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantage[index]

                actor_loss = -torch.min(surr1, surr2)
                actor_loss = torch.mean(actor_loss)

                #vloss-clip
                v_loss_unclipped = (v_s - v_target[index]).pow(2)
                v_clipped = vs[index] + torch.clamp(
                    v_s - vs[index],
                    -self.epsilon,
                    self.epsilon,
                )
                v_loss_clipped = (v_clipped - v_target[index]).pow(2)
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                dist_entropy = dist_entropy.mean()
                print('dist_entropy', to_numpy(dist_entropy))
                GLOBAL_LOGGER.get_tb_logger().add_scalar('entropy', to_numpy(dist_entropy),total_steps)  # 好像不是total_steps,先这样写着
                loss = actor_loss - self.entropy_coef * dist_entropy + 0.5 * v_loss
                #loss = loss.to(self.device)
                #loss.requires_grad_(True)   #用model.named_parameters()查看所有带命名的参数
                self.optimizer.zero_grad()
                #print('GCN', self.agent.GCN1.lin.weight)
                #print(self.optimizer.param_groups)
                loss.backward()

                #print('fc1', self.agent.fc1.weight, self.agent.fc1.weight.grad, self.agent.fc1.weight.requires_grad)
                #print('GCN', self.agent.GCN1.lin.weight)
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.2)
                self.optimizer.step()
                '''
                for parameters in self.optimizer.param_groups[0]['params']:
                    print('parameters', parameters[0])
                    print('parameters_grad', parameters.grad)
                '''

                GLOBAL_LOGGER.get_tb_logger().add_scalar('PPO.loss', to_numpy(loss), total_steps)
                print('PPO.loss', to_numpy(loss))
                '''
                actor_loss = actor_loss.to(self.device)
                # Update actor
                actor_loss = torch.mean(actor_loss)
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = (v_s - v_target[index]).pow(2).mean()
                critic_loss = critic_loss.to(self.device)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
                GLOBAL_LOGGER.get_tb_logger().add_scalar('PPO.loss_actor', to_numpy(actor_loss), total_steps)
                print('PPO.loss_actor', to_numpy(actor_loss))
                GLOBAL_LOGGER.get_tb_logger().add_scalar('PPO.loss_critic', to_numpy(critic_loss), total_steps)
                print('PPO.loss_critic', to_numpy(critic_loss))
                '''

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now
        GLOBAL_LOGGER.get_tb_logger().add_scalar('lr', lr_now, total_steps)
        print('lr', lr_now)


    def save(self, path: str, postfix: str, total_steps, env_reward):
        checkpoint = {
            'total_steps': total_steps,
            'env_reward': env_reward,
            'network': self.agent.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        torch.save(checkpoint, path + "agent_" + postfix + ".pt")