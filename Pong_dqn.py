import ale_py
import gymnasium as gym
import time
import threading
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy

class ParallelDqn(DQN):
    def __init__(self, policy, env, strategies, sync_frequency, **kwargs):
        self.strategies = np.array(strategies)
        self.sync_frequency = sync_frequency
        
        kwargs['exploration_fraction'] = 0.0
        kwargs['exploration_initial_eps'] = 0.0
        kwargs['exploration_final_eps'] = 0.0
        
        super().__init__(policy=policy, env=env, **kwargs)
        
        self.buffer_lock = threading.Lock()
        self.stop_training = False
        
        self.log_storage = None
        self.eval_env = None
        self.eval_freq = 0
        self.next_eval_step = 0
        self.start_time = 0
    def _excluded_save_params(self) -> list[str]: # Чтобы не было траблов с переходами между частотоми
        return super()._excluded_save_params() + ["buffer_lock", "eval_env", "log_storage", "stop_training"]

    def learn(self, total_timesteps, eval_env=None, eval_freq=None, log_storage=None, **kwargs):
        self.total_timesteps_limit = total_timesteps
        self.num_timesteps = 0
        self._n_updates = 0
        self.async_start_time = time.perf_counter() 
        self.stop_training = False 
        
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_storage = log_storage
        
        if self.eval_freq:
            self.next_eval_step = self.eval_freq
        
        self._setup_learn(total_timesteps, callback=None, reset_num_timesteps=True)

        actor_thread = threading.Thread(target=self._actor_loop)
        learner_thread = threading.Thread(target=self._learner_loop)
        actor_thread.start()
        learner_thread.start()
        
        actor_thread.join()
        self.stop_training = True
        learner_thread.join()
        return self
    
    def train_step(self, replay_data):
        with torch.set_grad_enabled(True):
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            with torch.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values)
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += 1

    def _actor_loop(self):
        obs = self.env.reset()
        self._last_obs = obs
        
        while self.num_timesteps < self.total_timesteps_limit and not self.stop_training:
            with torch.no_grad():
                unscaled_action, _ = self.policy.predict(self._last_obs, deterministic=True)
            
            for i in range(self.env.num_envs):
                if np.random.random() <= self.strategies[i]:
                    unscaled_action[i] = self.action_space.sample()
            
            new_obs, rewards, dones, infos = self.env.step(unscaled_action)
            self.num_timesteps += self.env.num_envs
            with self.buffer_lock:
                self.replay_buffer.add(self._last_obs, new_obs, unscaled_action, rewards, dones, infos)
            self._last_obs = new_obs
            
            if self.eval_env and self.num_timesteps >= self.next_eval_step:
                 self._run_evaluation()
                 self.next_eval_step += self.eval_freq
            
            time.sleep(0.003) # Эмпирически выяснено, моделька не успевает обновлять веса, поэтому ставим задержку

    def _learner_loop(self):
        while not self.stop_training:
            if self.replay_buffer.size() <= self.learning_starts: # ждем, чтобы из буфера можно было взять батч
                time.sleep(0.1)
                continue
            with self.buffer_lock:
                replay_data = self.replay_buffer.sample(self.batch_size)
            self.train_step(replay_data)
            if self._n_updates % self.sync_frequency == 0:
                self.q_net_target.load_state_dict(self.q_net.state_dict())



    def _run_evaluation(self):
        current_time = time.perf_counter() - self.async_start_time
        print(f"\n--- EVALUATION AT TIMESTEP {self.num_timesteps} (Updates: {self._n_updates}) ---")
        sys.stdout.flush()
        old_exploration_rate = self.exploration_rate
        self.exploration_rate = 0.05 
        mean_rew, _ = evaluate_policy(
            self,
            self.eval_env,
            n_eval_episodes=3,
            deterministic=False
        )
        self.exploration_rate = old_exploration_rate
        print(f"RESULT (with eps=0.05): Mean Reward: {mean_rew:.2f}")
        print("------------------------------------------\n")
        sys.stdout.flush()
        if self.log_storage is not None:
            self.log_storage['times'].append(current_time)
            self.log_storage['updates'].append(self._n_updates)
            self.log_storage['steps'].append(self.num_timesteps)
            self.log_storage['rewards'].append(mean_rew)

def plot_results(all_results):
    plt.style.use('ggplot')
    plots_config = [
        ('times', 'rewards', 'Reward vs Time', 'reward_vs_time.png', 'Time (s)', 'Mean Reward'),
        ('updates', 'rewards', 'Reward vs Updates', 'reward_vs_updates.png', 'Model Updates', 'Mean Reward'),
        ('steps', 'rewards', 'Reward vs Environment Steps', 'reward_vs_steps.png', 'Env Steps', 'Mean Reward'),
    ]

    for x_key, y_key, title, filename, xlabel, ylabel in plots_config:
        plt.figure(figsize=(10, 6))
        for freq, data in all_results.items():
            if len(data[x_key]) > 0:
                plt.plot(data[x_key], data[y_key], label=f'Sync Freq: {freq}', marker='o', markersize=3)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"График сохранен: {filename}")
        plt.close()

def train():
    TOTAL_TIMESTEPS = 2500000
    EVAL_FREQ = 25000
    
    DQN_HYPERPARAMS = {
      "learning_rate": 1e-4,
      "buffer_size": 1000000, 
      "learning_starts": 6250,
      "batch_size": 32,
      "gamma": 0.99,
      "optimize_memory_usage": False, 
      "train_freq": 1, 
      "gradient_steps": 1, 
    }

    eval_env = make_atari_env("PongNoFrameskip-v4", n_envs=1)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    all_results = {}

    frequencies = [10, 5, 1] 
    
    for freq in frequencies: 
        print(f"\n{'='*30}")
        print(f"STARTING ASYNC TRAIN: SYNC FREQ = {freq}")
        print(f"{'='*30}")
        
        train_env = make_atari_env("PongNoFrameskip-v4", n_envs=8, seed=42)
        train_env = VecFrameStack(train_env, n_stack=4)

        strategies = [0.05, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

        model = ParallelDqn(
            policy="CnnPolicy",
            env=train_env,
            strategies=strategies,
            sync_frequency=freq,
            verbose=0, 
            **DQN_HYPERPARAMS,
        )

        current_storage = {'times': [], 'updates': [], 'steps': [], 'rewards': []}
        
        # Передаем параметры напрямую в learn
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            eval_env=eval_env,
            eval_freq=EVAL_FREQ,
            log_storage=current_storage
        )
        
        model.save(f"dqn_pong_async_freq_{freq}")
        all_results[freq] = current_storage
        train_env.close()

    eval_env.close()
    print("\nОбучение завершено. Построение графиков...")
    
    plot_results(all_results)
    print("Готово.")

if __name__ == '__main__':
    train()
