import os
import pprint

import gym
import pygame
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import RainbowPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import NoisyLinear

from worstpractices import tolerate_extra_args, remedy
 
from evestop.generic import EVEEarlyStopping

global_config = {
    'algo': 'rainbow'
    }

assert torch.cuda.device_count() <= 1, 'Forgot to set CUDA_VISIBLE_DEVICES?'

early_stopping = EVEEarlyStopping(
    baseline=0, patience=1000, smoothing=0.75, min_delta=0, mode='max')

def save(obj, path):
    torch.save(obj, path, pickle_protocol=5)
    return path

def load(path):
    return torch.load(path)

def start_virtual_display():
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    #if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY"))==0:
    #    os.exec('../xvfb start')
    #    os.environ['DISPLAY'] = ':1'

make = tolerate_extra_args(gym.make)

@remedy(pygame.error, start_virtual_display)
def rainbow(
        seed=1626,
        eps_test=0.05,
        eps_train=0.1,
        buffer_size=10000,
        lr=1e-3,
        gamma=0.9,
        num_atoms=51,
        v_min=-10.,
        v_max=10.,
        noisy_std=0.1,
        n_step=3,
        target_update_freq=320,
        epoch=100,
        step_per_epoch=8000,
        step_per_collect=8,
        update_per_step=0.125,
        batch_size=64,
        hidden_sizes=[128, 128, 128, 128],
        training_num=8,
        test_num=100,
        logdir='log',
        render=0.,
        prioritized_replay=False,
        alpha=0.6,
        beta=0.4,
        beta_final=1.,
        resume=False,
        device='cuda',
        save_interval=4,
        task='FetalDiscrete-v0'):
    config = {**locals(), **global_config}
    logger = WandbLogger(project='fetal', 
                         save_interval=save_interval, 
                         config=config)
    writer = SummaryWriter(logdir)
    writer.add_text('config', str(config))
    logger.load(writer)

    video_path = os.path.join('videos', str(logger.wandb_run.id))

    env = make(task)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    # train_envs = make(task)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv(
        [lambda: make(task, partition='train') for _ in range(training_num)]
    )
    # test_envs = make(task)
    test_envs = DummyVectorEnv(
        [lambda: gym.wrappers.RecordVideo(make(task, partition='test'), 
                                          os.path.join(video_path, str(test_num))) 
         for _ in range(test_num)]
    )
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # model

    def noisy_linear(x, y):
        return NoisyLinear(x, y, noisy_std)

    net = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        device=device,
        softmax=True,
        num_atoms=num_atoms,
        dueling_param=({
            'linear_layer': noisy_linear
        }, {
            'linear_layer': noisy_linear
        })
    )
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    policy = RainbowPolicy(
        net,
        optim,
        gamma,
        num_atoms,
        v_min,
        v_max,
        n_step,
        target_update_freq=target_update_freq
    ).to(device)
    # buffer
    if prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            buffer_size,
            buffer_num=len(train_envs),
            alpha=alpha,
            beta=beta,
            weight_norm=True
        )
    else:
        buf = VectorReplayBuffer(buffer_size, buffer_num=len(train_envs))
    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=batch_size * training_num)
    # log
    log_path = os.path.join(logdir, str(logger.wandb_run.id))
    os.makedirs(log_path, exist_ok=True)
    ckpt_path = os.path.join(log_path, 'checkpoint.pth')
    policy_path = os.path.join(log_path, 'policy.pth')

    def stop_fn(mean_rewards):
        early_stopping.register(mean_rewards)
        return not early_stopping.proceed

    def train_fn(epoch, env_step):
        # eps annealing, just a demo
        if env_step <= 10000:
            policy.set_eps(eps_train)
        elif env_step <= 50000:
            eps = eps_train - (env_step - 10000) / \
                40000 * (0.9 * eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * eps_train)
        # beta annealing, just a demo
        if prioritized_replay:
            if env_step <= 10000:
                beta = beta
            elif env_step <= 50000:
                beta = beta - (env_step - 10000) / \
                    40000 * (beta - beta_final)
            else:
                beta = beta_final
            buf.set_beta(beta)

    def test_fn(epoch, env_step):
        policy.set_eps(eps_test)

    def save_policy_fn(policy):
        return save(policy.state_dict(), policy_path)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        return save(
            {
                'model': policy.state_dict(),
                'optim': optim.state_dict(),
                'buffer': train_collector.buffer
            }, ckpt_path
        )

    if resume:
        # load from existing checkpoint
        print(f'Loading agent under {log_path}')
        
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            policy.load_state_dict(checkpoint['model'])
            policy.optim.load_state_dict(checkpoint['optim'])
            train_collector.buffer = checkpoint['buffer']
            print('Successfully restore policy, optim and buffer')
        else:
            print('Failed to restore policy, optim and buffer.')

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        epoch,
        step_per_epoch,
        step_per_collect,
        test_num,
        batch_size,
        update_per_step=update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_policy_fn,
        logger=logger,
        resume_from_log=resume,
        save_checkpoint_fn=save_checkpoint_fn
    )
    assert stop_fn(result['best_reward'])