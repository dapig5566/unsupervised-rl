import gym as g
import time
# num_traj = 32
# env = g.make("CartPole-v1")
# ob_ph = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]])
# action_ph = tf.placeholder(tf.int32, [None])
# adv_ph = tf.placeholder(tf.float32, [None])
# x = tf.layers.dense(ob_ph, 32, tf.nn.relu)
# x = tf.layers.dense(x, 32, tf.nn.relu)
# x = tf.layers.dense(x, env.action_space.n)
# act_given_state = tf.squeeze(tf.random.multinomial(x, 1))#tf.squeeze(tf.argmax(x, axis=-1))
# x = tf.nn.softmax(x, axis=-1)
# N = tf.shape(x)[0]
# t = tf.transpose(tf.stack([tf.range(N), tf.squeeze(action_ph)]))
# log_prob = tf.log(tf.gather_nd(x, t))
# loss = -1.0 * tf.reduce_sum(log_prob * adv_ph) / num_traj
# update_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
#
# ss = tf.Session()
# ss.run(tf.global_variables_initializer())
#
#
# def compute_adv(rewards, gamma):
#     n = len(rewards)
#     rewards = [gamma**i * rewards[i] for i in range(n)]
#     rewards = [np.sum(rewards[i:]) / gamma**i for i in range(n)]
#     return np.array(rewards)
#
#
# def sample_trajs(env, size):
#     ob_batch = []
#     reward_batch = []
#     action_batch = []
#     for i in range(size):
#         ob = env.reset()
#         episode_ob = []
#         episode_reward = []
#         episode_action = []
#         time_step = 0
#         while time_step < env.spec.max_episode_steps:
#             episode_ob.append(ob)
#             action = ss.run(act_given_state, feed_dict={ob_ph: ob[None, :]})
#             ob, r, done, _ = env.step(action)
#             episode_action.append(action)
#             episode_reward.append(r)
#             time_step += 1
#         ob_batch.append(np.array(episode_ob))
#         reward_batch.append(np.array(episode_reward))
#         action_batch.append(np.array(episode_action))
#     print("episode reward: {}".format(np.sum(reward_batch) / size))
#     adv_batch = np.concatenate([compute_adv(rewards, 0.9) for rewards in reward_batch])
#     return np.concatenate(ob_batch), np.concatenate(action_batch), adv_batch
#
# for ep in range(100):
#     print("epoch {}:".format(ep))
#     obs, acs, ads = sample_trajs(env, num_traj)
#     ads = (ads - np.mean(ads)) / np.std(ads)
#     l, _ = ss.run([loss, update_op], feed_dict={ob_ph:obs, action_ph:acs, adv_ph:ads})
#     print("loss: {}".format(l))
# e = g.make("MountainCar-v0")
# e.set_goal(0.4)
# for i in range(10):
#     _ = e.reset()
#     while True:
#         e.render()
#         time.sleep(0.013)
#         _, _, done, _ = e.step(e.action_space.sample())
#         if done:
#             break
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
#
# def init_envs(env_name, nproc):
#     envs = [create_evn(env_name, seed) for seed in range(nproc)]
#     envs = SubprocVecEnv(envs)
#     return envs
#
# def create_evn(env_name, seed):
#     def _f():
#         e = g.make(env_name)
#         e.seed(seed)
#         return e
#     return _f
#
# num_workers = 3
#
# if __name__ == "__main__":
#
#     envs = init_envs("MountainCarContinuous-v0", num_workers)
#
#     obs = envs.reset()
#     envs.set_goal([[0.6, 0],[0.6, 0],[0.6, 0]])
#     for i in range(3000):
#         envs.render()
#         obs, rs, ds, _ = envs.step([envs.action_space.sample() for _ in range(num_workers)])
#         # print(obs)
#         # print(rs)
#         # print(ds)
#     envs.close()
import tensorflow as tf
import numpy as np
a=tf.ones([4, 2, 1])
b=tf.ones([4, 1, 2])*2
c = tf.matmul(a, b)

tf.InteractiveSession()
tf.global_variables_initializer().run()
print(c.eval(), np.shape(c.eval()))









