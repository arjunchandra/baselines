import tables
import argparse
import gym
import os
import numpy as np

from gym.monitoring import VideoRecorder

import baselines.common.tf_util as U

from baselines import deepq
from baselines.common.misc_util import (
    boolean_flag,
    SimpleMonitor,
)
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.deepq.experiments.atari.model import model, dueling_model





def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    parser.add_argument("--video", type=str, default=None, help="Path to mp4 file where the video of first episode will be recorded.")
    parser.add_argument("--store_transitions", type=str, default=None, help="Path to hdf5 file where transitions should be stored.")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    parser.add_argument("--scope", type=str, default="deepq", help="Tensorflow scope for agent´s network")

    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    env = SimpleMonitor(env)
    env = wrap_dqn(env)
    return env

def play(env, act, stochastic, video_path, store_transitions):
    max_steps=200000
    if store_transitions is not None:
        FILTERS = tables.Filters(complib='blosc', complevel=6)
        hdf5_file = tables.open_file(store_transitions, mode='w')
        obs_storage = hdf5_file.create_earray(hdf5_file.root, 'obs', tables.UInt8Atom(), shape=(0, 84, 84, 4), expectedrows = max_steps)
        act_storage = hdf5_file.create_earray(hdf5_file.root, 'act_rew_done', tables.FloatAtom(), shape=(0, 3), expectedrows = max_steps)

    num_episodes = 0
    #video_recorder = None
    #video_recorder = VideoRecorder(
    #    env, video_path, enabled=video_path is not None)
    obs = np.array(env.reset())
    n = 0
    while n < max_steps:
        n += 1
        if store_transitions is not None:
            obs_storage.append(obs[None])
        env.unwrapped.render()
        #video_recorder.capture_frame()
        action = act(obs[None], stochastic=stochastic)[0]
        obs, rew, done, info = env.step(action)
        obs = np.array(obs)
        if store_transitions is not None:
            act_storage.append([[action, rew, done]])

        if done:
            obs = env.reset()
            obs = np.array(obs)
        if len(info["rewards"]) > num_episodes:
            #if len(info["rewards"]) == 1 and video_recorder.enabled:
            #    # save video of first episode
            #    print("Saved video.")
            #    video_recorder.close()
            #    video_recorder.enabled = False
            print("Steps: ", n, " Reward: ", info["rewards"][-1])
            num_episodes = len(info["rewards"])

    if store_transitions is not None:
        hdf5_file.close()

if __name__ == '__main__':
    if True:
    #with U.make_session(4) as sess:
        args = parse_args()
        env = make_env(args.env)
    with U.make_session(4) as sess:
        act = deepq.build_act(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=dueling_model if args.dueling else model,
            num_actions=env.action_space.n,
            scope=args.scope)
        U.load_state(os.path.join(args.model_dir, "saved"))
        play(env, act, args.stochastic, args.video, args.store_transitions)

