import gym as g
import tensorflow as tf
import numpy as np
import argparse
import tensorflow_probability as tfp
import time
import matplotlib.pyplot as plt


def to_arc_state(obs):
    theta1 = [-np.arccos(state[0]) if state[1]<0 else np.arccos(state[0]) for state in obs[:, :2]]
    theta2 = [-np.arccos(state[0]) if state[1]<0 else np.arccos(state[0]) for state in obs[:, 2:4]]
    arc_state = np.concatenate([np.stack([theta1, theta2]).transpose()], axis=-1)
    return arc_state


def to_cos_state(sampled_states):
    cos_state = np.concatenate([
        np.cos(sampled_states[:, 0].reshape([-1, 1])),
        np.sin(sampled_states[:, 0].reshape([-1, 1])),
        np.cos(sampled_states[:, 1].reshape([-1, 1])),
        np.sin(sampled_states[:, 1].reshape([-1, 1])),
    ], axis=-1)

    return cos_state


def mc_to_full_state(positions): #MountainCar
    return np.concatenate([positions, np.zeros_like(positions)], axis=-1)


def preprocess_goals(goals, group_size):
    scale = np.array([0.9, 0.07])
    loc = np.array([-0.3, 0])
    goals = mc_to_full_state(goals)
    # goals = np.tanh(goals) * scale + loc
    goals = np.clip(goals, [-1.2, -0.07], [0.6, 0.07])
    return goals.reshape([-1, group_size, goals.shape[-1]])


def separate_parallel(arr):
    axes = list(range(len(arr.shape)))
    axes[:2] = [1, 0]
    return [i for i in np.transpose(arr, axes)]


def sample_trajectory(e, batch_size, agent=None, session=None, goals=None, render=False, min_l=512, max_episode_l=200):
    if goals is not None:
        assert goals.shape[0] == batch_size, "number of goals must equal to the batch size if there are goals."

    ob_batch = []
    reward_batch = []
    action_batch = []
    success_count = 0
    total_length = 0
    i=0
    num_parallel = None
    while total_length < min_l or i < batch_size:
        ob = e.reset()
        num_parallel = len(ob)
        if goals is not None:
            current_goal = goals[i]
            e.set_goal(current_goal)
        else:
            e.set_goal([0.6, 0])

        current_obs = []
        current_rewards = []
        current_actions = []
        all_done = np.array([0 for _ in range(num_parallel)])
        while True:
            if goals is not None:
                ob = np.concatenate([ob, current_goal], axis=-1)
            current_obs.append(ob)
            if render:
                e.render()
            if agent is None:
                action = [e.action_space.sample() for _ in range(num_parallel)]
            else:
                action = agent.act(ob, session)

            ob, reward, done, _ = e.step(action)
            if goals is not None:
                # norm = np.sum(np.abs(current_goal - ob[:4]))
                # norm = np.minimum(norm, 1.5)
                norm = np.sqrt(np.sum(np.square(current_goal - ob), axis=-1))
                reward = -np.squeeze(norm)
                all_done += (norm <= 0.005).astype("int")
            current_rewards.append(reward)
            current_actions.append(action)
            if np.all(all_done > 0) or len(current_obs) >= max_episode_l:
                success_count += np.sum(all_done > 0)
                break
        total_length += len(current_obs) * num_parallel
        ob_batch.extend(separate_parallel(np.array(current_obs)))
        reward_batch.extend(separate_parallel(np.array(current_rewards)))
        action_batch.extend(separate_parallel(np.array(current_actions)))
        i += 1

    return ob_batch, reward_batch, action_batch, success_count / (batch_size * num_parallel)


def get_advantage(rewards, gamma):
    rewards = [gamma**i * r for i, r in enumerate(rewards)]
    rewards = [np.sum(rewards[i:]) / gamma**i for i in range(len(rewards))]
    return np.array(rewards)


class MultivariateGaussian:

    def __init__(self, sample_size, obs_dim, goal_batch_size, lr=0.1):
        self.lr = lr
        self.obs_dim = obs_dim
        self.sample_size = sample_size
        self.input_samples = tf.placeholder(tf.float32, [None, obs_dim])
        self.mean, self.cov = self.create_params()
        self.goal_batch_size = goal_batch_size
        self.distribution = tfp.distributions.MultivariateNormalFullCovariance(loc=self.mean,
                                                                          covariance_matrix=self.cov)
        self.sample_from_dist = self.distribution.sample([self.goal_batch_size])
        probs = 1 / self.distribution.prob(self.input_samples)
        self.sample_weights = probs / tf.reduce_sum(probs)
        self.update_ops = self.build_update_op()

    def create_params(self):
        with tf.variable_scope("multi_gauss"):
            mean = tf.get_variable("mean", shape=[self.obs_dim], dtype=tf.float32, initializer=tf.keras.initializers.zeros)
            cov = tf.get_variable("cov", shape=[self.obs_dim, self.obs_dim], dtype=tf.float32, initializer=tf.keras.initializers.identity)
        return mean, cov

    def compute_exp_vec(self):
        mean = tf.reduce_sum(self.input_samples * tf.reshape(self.sample_weights, [-1, 1]), axis=0)
        return mean

    def compute_cov_mat(self, mean):
        shifted_samples = self.input_samples - mean
        cov = tf.reduce_sum(tf.matmul(tf.expand_dims(shifted_samples, axis=1),
                                      tf.expand_dims(shifted_samples, axis=2)) *
                            tf.reshape(self.sample_weights, [-1, 1, 1]), axis=0)
        return cov

    def build_update_op(self):
        mean = self.compute_exp_vec()
        cov = self.compute_cov_mat(mean)
        op1 = tf.assign(self.mean, self.mean * (1-self.lr) + mean * self.lr)
        op2 = tf.assign(self.cov, self.cov * (1-self.lr) + cov * self.lr)
        return [op1, op2]

    def fit(self, obs, session=None):
        feed_dict = {self.input_samples: obs}
        _ = session.run(self.update_ops, feed_dict=feed_dict)

    def sample(self, session=None):
        return self.sample_from_dist.eval(session=session)


def build_mlp(input, layers):
    x = input
    for i, j in zip(layers, range(len(layers))):
        if j == len(layers)-1:
            x = tf.layers.dense(x, i)
        else:
            x = tf.layers.dense(x, i, activation=tf.nn.relu)
    return x


class Agent:

    def __init__(self, state_dim, action_dim, discrete=True):
        # self.scale = tf.ones([action_dim]) * 0.6
        self.obs_goal_ph = tf.placeholder(tf.float32, [None, state_dim])
        self.advantage_ph = tf.placeholder(tf.float32, [None])
        if discrete:
            self.action_ph = tf.placeholder(tf.int32, [None])
            self.model_output = build_mlp(self.obs_goal_ph, [32, 32, 32, action_dim])
        else:
            self.action_ph = tf.placeholder(tf.float32, [None, action_dim])
            self.model_output = tf.reshape(build_mlp(self.obs_goal_ph, [32, 32, 32, action_dim * 2]), [-1, 2, action_dim])
            # self.model_output = build_mlp(self.obs_goal_ph, [32, 32, 32, action_dim])

        self.discrete = discrete
        N = tf.shape(self.model_output)[0]
        if discrete:
            self.sample_action = tf.squeeze(tf.random.multinomial(self.model_output, N))
        else:
            # self.scale = tf.ones([action_dim], dtype=tf.float32)
            self.sample_action = tf.random.normal([N, action_dim]) * self.model_output[:, 1, :] + self.model_output[:, 0, :]
            # self.sample_action = tf.random.normal([N, action_dim]) * self.scale + self.model_output
        self.train_op = self.build_train_op()

    def build_train_op(self):
        if self.discrete:
            N = tf.shape(self.action_ph)[0]
            idx = tf.transpose(tf.stack([tf.range(0, N), self.action_ph]))
            log_probs = tf.log(tf.gather_nd(tf.nn.softmax(self.model_output, axis=1), idx))
        else:
            distribution = tfp.distributions.MultivariateNormalDiag(self.model_output[:, 0, :], tf.exp(self.model_output[:, 1, :]))
            # distribution = tfp.distributions.MultivariateNormalDiag(self.model_output, self.scale)
            log_probs = distribution.log_prob(self.action_ph)
        self.loss = -tf.reduce_sum(log_probs * self.advantage_ph) / tf.to_float(num_trajs)
        train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)
        return train_op

    def train(self, obs_goals, actions, advs, session=None):
        feed_dict = {self.obs_goal_ph: obs_goals, self.action_ph: actions, self.advantage_ph: advs}
        loss, _ = session.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def act(self, obs_goals, session=None):
        feed_dict = {self.obs_goal_ph: obs_goals}
        action = session.run(self.sample_action, feed_dict=feed_dict)
        return action

from baselines.baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def init_envs(env_name, nproc):
    envs = [create_evn(env_name, seed) for seed in range(nproc)]
    envs = SubprocVecEnv(envs)
    return envs

def create_evn(env_name, seed):
    def _f():
        e = g.make(env_name)
        e.seed(seed)
        return e
    return _f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", default=512)
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--lr", default=0.1)
    parser.add_argument("--num_workers", default=5)
    parser.add_argument("--env_name", default="MountainCarContinuous-v0")

    args = parser.parse_args()
    iters = 10
    num_workers = args.num_workers
    num_trajs = num_workers * args.batch_size
    success_rates = []
    env = init_envs(args.env_name, num_workers)
    discrete = isinstance(g.make(args.env_name).action_space, g.spaces.Discrete)
    gaussian = MultivariateGaussian(sample_size=args.sample_size, obs_dim=1, goal_batch_size=args.batch_size * num_workers, lr=args.lr)
    agent = Agent(env.observation_space.shape[0] + env.observation_space.shape[0], g.make(args.env_name).action_space.n if discrete else g.make(args.env_name).action_space.shape[0], discrete)
    session = tf.Session()

    session.run(tf.global_variables_initializer())
    print("Fitting initial distribution...")
    obs, _, _, _ = sample_trajectory(env, 2)
    obs = np.concatenate(obs)
    # idx = np.random.choice(obs.shape[0], np.minimum(args.sample_size, obs.shape[0]).astype("int"), False)
    # gaussian.fit(obs[idx, :1], session=session)
    gaussian.fit(obs[:, :1], session=session)

    for ep in range(args.epochs):
        print()
        print("="*30)
        print("epoch {}".format(ep+1))
        goals = preprocess_goals(gaussian.sample(session=session), num_workers)
        print("performing goal achieving...")
        ob_batch = None
        for i in range(iters):
            ob_batch, reward_batch, \
            action_batch, success_rate = sample_trajectory(env, args.batch_size, agent,
                                                           goals=goals, render=False, session=session, min_l=args.sample_size)#(ep%10 == 0)
            baseline = np.sum(np.concatenate(reward_batch)) / num_trajs
            ob_batch = np.concatenate(ob_batch)
            advantage_batch = np.concatenate([get_advantage(i, 0.9) for i in reward_batch]) - baseline
            advantage_batch = (advantage_batch - np.mean(advantage_batch)) / np.std(advantage_batch)
            action_batch = np.concatenate(action_batch)
            loss = agent.train(ob_batch, action_batch, advantage_batch, session=session)

            if (i+1) % 5 == 0:
                print()
                print("*" * 15)
                print("total length: {}".format(ob_batch.shape[0]))
                print("iteration {}".format(i+1))
                print("success rate: {}".format(success_rate))
                success_rates.append(success_rate)
                print("average episode reward: {}".format(baseline))
                print("epoch loss: {}".format(loss))
                print("*"*15)

        print("fitting new distribution...")
        # idx = np.random.choice(ob_batch.shape[0], np.minimum(args.sample_size, ob_batch.shape[0]).astype("int"), False)
        # gaussian.fit(ob_batch[idx, :env.observation_space.shape[0]-1], session=session)
        gaussian.fit(ob_batch[:, :env.observation_space.shape[0]-1], session=session)
        print("="*30)

    plt.plot(success_rates)
    plt.show()
    for i in range(5):
        goals = preprocess_goals(gaussian.sample(session=session), num_workers)
        ob_batch, reward_batch, \
        action_batch, success_rate = sample_trajectory(env, args.batch_size, agent,
                                                       goals=goals, render=True, session=session, min_l=args.sample_size)
        print("final test success rate: {}".format(success_rate))







