#  DRL-5G-Scheduler; Author: Zhouyou Gu (zhouyou.gu@sydney.edu.au);
#  Supervisors: Wibowo Hardjawana; Branka Vucetic;
#  This project is developed at Centre for IoT and Telecommunications at The University of Sydney,
#  under a project directly funded by Telstra Corporation Ltd., titled
#  ”Development of an Open Programmable Scheduler for LTE Networks”, from 2018 to 2019.
#  Reference: Z. Gu, C. She, W. Hardjawana, S. Lumb, D. McKechnie, T. Essery, and B. Vucetic,
#   “Knowledge-assisted deep reinforcement learning in 5G scheduler design:
#  From theoretical framework to implementation,” IEEE JSAC., to appear, 2021

import os
from multiprocessing import Process

from pytorch.tb_logger import TBScalarToCSV

path = os.path.dirname(os.path.realpath(__file__))
# path_data = os.path.join(path, 'tb-data')
path_data = r"/home/cuichuankai/GCN-DL/5.PPO-continuous/data"
dirs = next(os.walk(path_data))[1]  # os.walk return a generator,返回的是path_data下的目录，因为是从元组里选了[1]
# os.walk(path_data)
# for root, dirs, files in os.walk(path_data):
    # print(root, dirs, files)
# dirs = path_data

scalar_list = []


scalar = 'action'
scalar_list.append(scalar)
scalar = 'reward'
scalar_list.append(scalar)
scalar = 'avg_reward'
scalar_list.append(scalar)
scalar = 'new_total_avg_reward'
scalar_list.append(scalar)
scalar = 'old_total_avg_reward'
scalar_list.append(scalar)

def process_env(path, dir):
    p = os.path.join(path, dir)  #路径拼接
    TBScalarToCSV(p, p, scalar_list)


p_list = []
for d in dirs:
    p = Process(target=process_env, args=(path_data, d))  #比如有四组跑完的数据，那么创建四个process，放到list里
    p_list.append(p)

for p in p_list:  #开四个进程
    p.start()

for p in p_list:
    p.join()
