# python train.py --num_env=16 --alg=ddpg --env=ActiveTrack-v0 --num_timesteps=1000000 --save_path=./policymy/Activetest/ddpgTest --save_video_interval=1 --log_path=./logs/ddpg/ActiveTrack-1e5_0
# python train.py --num_env=8 --alg=ddpg --env=ActiveTrack-v0 --num_timesteps=10000000 --save_path=./policymy/Activetest/ddpgTest --save_video_interval=10 --log_path=./logs/ddpg/ActiveTrack-1e5_0
# python train.py --num_env=9 --alg=her --env=NeedlePick-v0 --num_timesteps=1e5 --policy_save_interval=10000 --demo_file=./surrol/data/demo/data_NeedlePick-v0_random_400.npz --save_path=./policymy/her/NeedlePick-demo1e5_0 --bc_loss=1 --q_filter=1 --num_demo=400 --demo_batch_size=128 --prm_loss_weight=0.001 --aux_loss_weight=0.0078 --n_cycles=20 --batch_size=1024 --random_eps=0.1 --noise_eps=0.1 --log_path=./logs/her/NeedlePick-demo1e5_0 --save_video_interval=10000

import sys
import os
import numpy as np
_old_load = np.load
np.load = lambda *a, **k: _old_load(*a, allow_pickle=True, **k)
import surrol.gym
from baselines.run import main
import baselines.common.tf_util as U

import baselines.her

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 允许显存按需增长，而不是一次占满
U.get_session(config=config)

print("import successful")
model_path = r"./policymy/NeedlePick-demo1e5_0/herTest"

if __name__ == '__main__':
    try:
        main(sys.argv)
        U.save_variables(model_path)
    except Exception as e:
        print(e)
        U.save_variables(model_path)