#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle, os, gym, load_policy, tf_util
import tensorflow as tf
import numpy as np
import IPython

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-m', "--max_timesteps", type=int)
    parser.add_argument('-n', '--num_rollouts', type=int, default=50,
                        help='Number of expert roll outs')
    parser.add_argument('-s', "--std_expert", type=float, default=0)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('-su', "--suffix", type=str, default="")
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                if args.std_expert > 0:
                    action = action + np.random.normal(scale=args.std_expert, size=action.shape)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        if args.save:
            expert_data = {'observations': np.array(observations),
                           'actions': np.array(actions),
                           'returns': returns,
                           'mean_return': np.mean(returns),
                           'std_return': np.std(returns)
                           }

            output_filename = os.path.join('expert_data', "{}{}.pkl".format(args.envname, args.suffix))
            with file(output_filename, 'w') as f:
                pickle.dump(expert_data, f)

if __name__ == '__main__':
    main()
