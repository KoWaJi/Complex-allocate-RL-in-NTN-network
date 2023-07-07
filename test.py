import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo import PPO_continuous
from Sim_env.env_config import *
from Sim_env.env import ENV
from tb_logger import GLOBAL_LOGGER
import os

def evaluate_policy(env, agent, evaluate_num):
    sum_reward = 0
    total_step = 0
    evaluate_reward_old = 0
    while total_step < evaluate_num:
        s = env.reset()
        done = False
        print('a new episode')
        while not done:
            total_step += 1
            print('total_step', total_step)
            true_ue = len(s) / 2
            x = create_x(s)
            edge_index = create_edge_index(s)
            a = agent.evaluate(x, edge_index)  # We use the deterministic policy during the evaluating
            action = a
            s_, r, _, done = env.step(action)
            active_ue = env.active_ue
            reward = np.sum(r)
            evaluate_reward_old = 0.99 * evaluate_reward_old + 0.01 * reward
            sum_reward += reward
            evaluate_reward_new = sum_reward / total_step
            GLOBAL_LOGGER.get_tb_logger().add_scalar('reward_old', evaluate_reward_old, total_step)
            GLOBAL_LOGGER.get_tb_logger().add_scalar('reward_new', evaluate_reward_new, total_step)
            GLOBAL_LOGGER.get_tb_logger().add_scalar('active_ue', active_ue, total_step)
            s = s_

def create_x(s):
    s = list(s)
    s.insert(int(len(s) / 2), 0)
    s.insert(0, 0)  # 插入基站
    true_ue = int(len(s) / 2)
    hol = torch.tensor(s[:true_ue]).reshape(true_ue, 1)
    n_rb = torch.tensor(s[true_ue:]).reshape(true_ue, 1)
    x = torch.cat((hol, n_rb), dim=1)
    return x.float().to(torch.device('cuda:2'))


def create_edge_index(s):
    true_ue = int(len(s) / 2)
    d = [true_ue]
    for i in range(true_ue):
        d.append(1)
    D = np.diag(d)
    A = np.zeros((true_ue + 1, true_ue + 1))
    for i in range(true_ue):
        A[0][i + 1] = 1
        A[i + 1][0] = 1
    edge_index = D + A
    return torch.tensor(edge_index).float().to(torch.device('cuda:2'))

def main(args, env_name, config, number):
    env = ENV(env_name, config)
    env_evaluate = ENV(env_name, config)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    '''
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    '''
    #np.random.seed(seed)
    #torch.manual_seed(seed)

    args.max_action = 2
    args.max_episode_steps = 200  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 10000  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    agent = PPO_continuous(args)
    if args.continue_train:
        checkpoint = torch.load('data/0.3/agent_final.pt')
        agent.agent.load_state_dict(checkpoint['network'], strict=True)
    # Build a tensorboard
    '''
    writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))
    '''
    evaluate_policy(env, agent, evaluate_num)
    print('overload time', env.getoverload_time())


    print('finish')

    '''
            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm, evaluate_num)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name, number, seed), np.array(evaluate_rewards))
                    test = np.load('./data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name, number, seed), np.array(evaluate_rewards))
                    print(test)
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(150000), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=int(2048), help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--embedding_width1", type=int, default=int(8), help=" The embedding width of GCN output")
    parser.add_argument("--embedding_width2", type=int, default=int(16), help=" The embedding width of GCN output")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of agent")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=3, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--continue_train", type=float, default=True, help="justice continue train or not")

    log_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    folder_name = "test data"
    experiment_name = "PPO-continuous"
    GLOBAL_LOGGER.set_log_path(log_path, folder_name, experiment_name)
    scalar_list = []
    args = parser.parse_args()
    scalar = 'active_ue'
    scalar_list.append(scalar)
    scalar = 'reward_old'
    scalar_list.append(scalar)
    scalar = 'reward_new'
    scalar_list.append(scalar)
    GLOBAL_LOGGER.get_tb_logger().set_scalar_filter(scalar_list)


    env_c = env_config_helper()
    env_c.reload_config()
    env_name = ['simulation environment']
    env_index = 0
    main(args, env_name=env_name[env_index], config=env_c.sim_env_config, number=1)
    GLOBAL_LOGGER.close_logger()
