import gym
import gym_bandits
import argparse
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
from utils import *
from policy import LSTMPolicy
from a2c import A2C

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_env', type=str, default="BanditTenArmedRandomRandom-v0", help='env for meta-training')
	parser.add_argument('--train_eps', type=int, default=100, help='training episodes per trial')
	parser.add_argument('--train_trial_n', type=int, default=1000, help='number of trials during training')
	parser.add_argument('--seed', type=int, default=1, help='experiment seed')

	# Training Hyperparameters
	parser.add_argument('--hidden', type=int, default=48, help='hidden layer dimensions')
	parser.add_argument('--gamma', type=float, default=0.8, help='discount factor')
	args = parser.parse_args()

	x, y, e = [], [], []
	for trial in range(1,args.train_trial_n+1):
		env = gym.make(args.train_env)
		env._seed(args.seed)
		env.reset()
		
		# initialize algorithm at first iteration
		if trial == 1:
			action_dim = env.action_space.n
			input_dim = 3
			algo = A2C(
				session=get_session(),
			    policy_cls=LSTMPolicy,
			    input_dim=input_dim,
			    hidden_dim=args.hidden,
			    action_dim=action_dim,
				scope='a2c')
		algo.reset()
		"""
		what does the env.unwrapped do exactly?

		https://discuss.pytorch.org/t/in-the-official-q-learning-example-what-does-the-env-unwrapped-do-exactly/28695

		there is a core super class called gym.Env and there are other sub classes of this to implement different environments (CartPoleEnv, MountainCarEnv etc). This unwrapped property is used to get the underlying gym.Env object from other environments.
		"""

		save_iter = args.train_trial_n // 20
		tot_returns = []
		prop_reward = []
		tot_regret = []
		tot_subopt = []

		ep_X, ep_R, ep_A, ep_V, ep_D = [], [], [], [], []
		track_R = 0 
		track_regret = np.max(env.unwrapped.p_dist) * args.train_eps
		best_action = np.argmax(env.unwrapped.p_dist) 
		num_suboptimal = 0
		action_hist = np.zeros(env.action_space.n)

		action = 0
		rew = 0
		# begin a trial
		for ep in range(args.train_eps):
			# run policy
			#print(action,rew, ep)
			algo_input = np.array([action,rew,ep])
			#print(algo_input)
			if len(algo_input.shape) <= 1:
				algo_input = algo_input[None]
			action, value = algo.get_actions(algo_input)
			new_obs, rew, done, info = env.step(action)
			track_R += rew
			num_suboptimal += int(action != best_action)
			action_hist[action] += 1
			if ep == 0:
				ep_X = algo_input
			else:
				ep_X = np.concatenate([ep_X,algo_input],axis=0)
			ep_A.append(action)
			ep_V.append(value)
			ep_R.append(rew)
			ep_D.append(done)

		# update policy
		ep_X = np.asarray(ep_X, dtype=np.float32)
		ep_R = np.asarray(ep_R, dtype=np.float32)
		ep_A = np.asarray(ep_A, dtype=np.int32)
		ep_V = np.squeeze(np.asarray(ep_V, dtype=np.float32))
		ep_D = np.asarray(ep_D, dtype=np.float32)
		last_value = value

		if ep_D[-1] == 0:
			disc_rew = discount_with_dones(ep_R.to_list() + [np.squeeze(last_value)], ep_D.to_list() + [0], args.gamma)[:-1]
		else:
			disc_rew = discount_with_dones(ep_R.tolist(), ep_D.tolist(), args.gamma)
		ep_adv = disc_rew - ep_V
		prop_reward.append(track_R/track_regret)
		track_regret -= track_R

		train_info = algo.train(ep_X=ep_X, ep_A=ep_A, ep_R=ep_R, ep_adv=ep_adv)
		tot_returns.append(track_R)
		tot_regret.append(track_regret)
		tot_subopt.append(num_suboptimal)

		if trial % save_iter == 0 and trial != 0:
			print("Episode: {}".format(trial))
			print("MeanReward: {}".format(np.mean(tot_returns[-save_iter:])))
			print("StdReward: {}".format(np.std(tot_returns[-save_iter:])))
			print("MeanRegret: {}".format(np.mean(tot_regret[-save_iter:])))
			print("StdRegret: {}".format(np.std(tot_regret[-save_iter:])))
			print("NumSuboptimal: {}".format(np.mean(tot_subopt[-save_iter:])))
			cur_y = np.mean(prop_reward[-save_iter:])
			cur_e = np.std(prop_reward[-save_iter:])
			x.append(trial)
			y.append(cur_y)
			e.append(cur_e)
			print("MeanPropReward: {}".format(cur_y))
			print("StdPropReward: {}".format(cur_e))

	x = np.asarray(x,dtype=np.int)
	y = np.asarray(y,dtype=np.float32)
	e = np.asarray(e,dtype=np.float32)
	# plt.errorbar(x, y, e)
	# plt.show()

	# database 
	db = {} 
	db['x'] = x 
	db['y'] = y
	db['e'] = e 
	  
	file_name = args.train_env[:-3] + str(args.train_trial_n)
	pickle.dump( db, open(file_name+".p","wb"))


if __name__=='__main__':
	main()
