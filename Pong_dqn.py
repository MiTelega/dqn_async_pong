import ale_py
import gymnasium as gym
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import copy
import traceback

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

def actor_process(rank, strategy_eps, queue, weight_dict, sync_freq, stop_event, env_id, seed, policy_kwargs):
    torch.set_num_threads(1)
    try:
        env = make_atari_env(env_id, n_envs=1, seed=seed + rank)
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
        
        policy_class = DQN.policy_aliases["CnnPolicy"]
        policy = policy_class(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: 0.0,
            **policy_kwargs
        )
        policy.eval()
        
        last_obs = env.reset()
        last_sync_update = -1

        while not stop_event.is_set():
            if mp.parent_process() is None or not mp.parent_process().is_alive():
                break

            if 'update_count' in weight_dict:
                learner_update_count = weight_dict['update_count']
                if learner_update_count > last_sync_update:
                    try:
                        policy.q_net.load_state_dict(weight_dict['state_dict'])
                        last_sync_update = learner_update_count
                    except Exception:
                        pass

            if np.random.random() <= strategy_eps:
                action = np.array([env.action_space.sample()])
            else:
                with torch.no_grad():
                    action, _ = policy.predict(last_obs, deterministic=True)

            new_obs, reward, done, info = env.step(action)
            transition = (last_obs, action, reward, new_obs, done, info)
            
            try:
                queue.put(transition, timeout=1.0)
            except:
                pass
            last_obs = new_obs
            
    except Exception:
        pass
    finally:
        try: 
            env.close()
        except: 
            pass

class ParallelDqn(DQN):
    def __init__(self, policy, env, strategies, sync_frequency, **kwargs):
        self.strategies = np.array(strategies)
        self.sync_frequency = sync_frequency
        kwargs['exploration_fraction'] = 0.0
        kwargs['exploration_initial_eps'] = 0.0
        kwargs['exploration_final_eps'] = 0.0
        
        super().__init__(policy=policy, env=env, **kwargs)
        
        self.log_storage = None
        self.eval_env = None
        self.eval_freq = 0
        self.next_eval_step = 0
        self.async_start_time = 0
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Learner device: {self.train_device}")

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["eval_env", "log_storage", "strategies", "sync_frequency"]

    def learn(self, total_timesteps, eval_env=None, eval_freq=None, log_storage=None, **kwargs):
        self.total_timesteps_limit = total_timesteps
        self.num_timesteps = 0
        self._n_updates = 0
        self.async_start_time = time.perf_counter()
        
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_storage = log_storage
        if self.eval_freq: self.next_eval_step = self.eval_freq
            
        self._setup_learn(total_timesteps, callback=None, reset_num_timesteps=True)
        
        self.policy.to(self.train_device)
        self.q_net.to(self.train_device)
        self.q_net_target.to(self.train_device)

        mp_manager = mp.Manager()
        shared_weights = mp_manager.dict()
        shared_weights['state_dict'] = {k: v.cpu() for k, v in self.q_net.state_dict().items()}
        shared_weights['update_count'] = 0
        
        data_queue = mp.Queue(maxsize=1024)
        stop_event = mp.Event()
        p_kwargs = self.policy_kwargs if self.policy_kwargs else {}

        processes = []
        n_actors = len(self.strategies)
        print(f"Запуск {n_actors} процессов-акторов...")
        
        for i in range(n_actors):
            p = mp.Process(
                target=actor_process,
                args=(i, self.strategies[i], data_queue, shared_weights, self.sync_frequency, stop_event, "PongNoFrameskip-v4", 42 + i, p_kwargs)
            )
            p.daemon = True 
            p.start()
            processes.append(p)

        print("Learner loop started.")
        steps_since_target_update = 0
        
        try:
            while self.num_timesteps < total_timesteps:
                try:
                    transition = data_queue.get(timeout=0.5)
                except:
                    if not any(p.is_alive() for p in processes): 
                        break
                    continue
                
                obs, action, reward, next_obs, done, infos = transition
                self.replay_buffer.add(obs, next_obs, action, reward, done, infos)
                self.num_timesteps += 1
                steps_since_target_update += 1

                if self.num_timesteps > self.learning_starts and self.num_timesteps % 4 == 0:
                    self.train_step()
                    if self._n_updates % self.sync_frequency == 0:
                        shared_weights['state_dict'] = {k: v.cpu() for k, v in self.q_net.state_dict().items()}
                        shared_weights['update_count'] = self._n_updates

                if steps_since_target_update >= 1000:
                    self.q_net_target.load_state_dict(self.q_net.state_dict())
                    steps_since_target_update = 0

                if self.eval_env and self.num_timesteps >= self.next_eval_step:
                    self._run_evaluation()
                    self.next_eval_step += self.eval_freq

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            print("Stopping actors...")
            stop_event.set()
            time.sleep(0.1)
            try:
                while not data_queue.empty(): data_queue.get_nowait()
            except: 
                pass
            for p in processes: 
                if p.is_alive(): 
                    p.terminate()
            for p in processes: 
                p.join(timeout=0.5)
            for p in processes: 
                if p.is_alive(): 
                    p.kill()
            print("Cleanup done.")
        return self

    def train_step(self):
        if self.replay_buffer.size() < self.batch_size: return
        replay_data = self.replay_buffer.sample(self.batch_size)
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

    def _run_evaluation(self):
        current_time = time.perf_counter() - self.async_start_time
        print(f"Async Eval ({self.num_timesteps}): ", end="")
        orig_eps = self.exploration_rate
        self.exploration_rate = 0.05
        mean_rew, _ = evaluate_policy(self, self.eval_env, n_eval_episodes=3, deterministic=False)
        self.exploration_rate = orig_eps
        print(f"{mean_rew:.2f}")
        if self.log_storage is not None:
            self.log_storage['times'].append(current_time)
            self.log_storage['updates'].append(self._n_updates)
            self.log_storage['steps'].append(self.num_timesteps)
            self.log_storage['rewards'].append(mean_rew)



class VanillaLoggingCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, log_storage):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_storage = log_storage
        self.next_eval = 0
        self.start_time = 0

    def _on_training_start(self):
        self.start_time = time.perf_counter()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            print(f"Standard Eval ({self.num_timesteps}): ", end="")
            mean_rew, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=3, deterministic=True)
            
            print(f"{mean_rew:.2f}")
            
            elapsed = time.perf_counter() - self.start_time
            approx_updates = int(self.num_timesteps / 4)
            
            self.log_storage['times'].append(elapsed)
            self.log_storage['updates'].append(approx_updates)
            self.log_storage['steps'].append(self.num_timesteps)
            self.log_storage['rewards'].append(mean_rew)
            
            self.next_eval += self.eval_freq
        return True

def plot_results(all_results):
    plt.style.use('ggplot')
    plots_config = [
        ('times', 'rewards', 'Reward vs Time', 'reward_vs_time.png', 'Time (s)', 'Mean Reward'),
        ('updates', 'rewards', 'Reward vs Updates', 'reward_vs_updates.png', 'Model Updates', 'Mean Reward'),
        ('steps', 'rewards', 'Reward vs Environment Steps', 'reward_vs_steps.png', 'Env Steps', 'Mean Reward'),
    ]

    for x_key, y_key, title, filename, xlabel, ylabel in plots_config:
        plt.figure(figsize=(10, 6))
        for label, data in all_results.items():
            if len(data[x_key]) > 0:
                style = '--' if 'Standard' in str(label) else '-'
                plt.plot(data[x_key], data[y_key], label=f'{label}', linestyle=style, marker='o', markersize=3)
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
    try: 
        mp.set_start_method('spawn', force=True)
    except: 
        pass
    torch.set_num_threads(4) 

    TOTAL_TIMESTEPS = 3000
    EVAL_FREQ = 250
    
    DQN_HYPERPARAMS_ASYNC = {
      "learning_rate": 1e-4,
      "buffer_size": 1000000, 
      "learning_starts": 100, 
      "batch_size": 32,
      "gamma": 0.99,
      "optimize_memory_usage": False, 
      "target_update_interval": 10000000,
      "train_freq": 10000000,
      "gradient_steps": 1,
    }

    DQN_HYPERPARAMS_STD = {
      "learning_rate": 1e-4,
      "buffer_size": 1000000,
      "learning_starts": 100,
      "batch_size": 32,
      "gamma": 0.99,
      "optimize_memory_usage": False,
      "target_update_interval": 1000, 
      "train_freq": 4,
      "gradient_steps": 1,
      "exploration_fraction": 0.1,
      "exploration_final_eps": 0.01
    }

    eval_env = make_atari_env("PongNoFrameskip-v4", n_envs=1)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)
    
    all_results = {}
    
    frequencies = [10, 5, 1] 
    
    dummy_env = make_atari_env("PongNoFrameskip-v4", n_envs=1)
    dummy_env = VecFrameStack(dummy_env, n_stack=4)
    dummy_env = VecTransposeImage(dummy_env)

    for freq in frequencies: 
        print(f"\n{'='*30}")
        print(f"STARTING ASYNC TRAIN: SYNC FREQ = {freq}")
        print(f"{'='*30}")

        strategies = [0.05, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

        model = ParallelDqn(
            policy="CnnPolicy",
            env=dummy_env,
            strategies=strategies,
            sync_frequency=freq,
            verbose=0, 
            device="auto",
            **DQN_HYPERPARAMS_ASYNC,
        )

        current_storage = {'times': [], 'updates': [], 'steps': [], 'rewards': []}
        
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            eval_env=eval_env,
            eval_freq=EVAL_FREQ,
            log_storage=current_storage
        )
        
        model.save(f"dqn_pong_async_freq_{freq}")
        all_results[f"Async Freq {freq}"] = current_storage
        del model
        torch.cuda.empty_cache()

    dummy_env.close()

    print(f"\n{'='*30}")
    print(f"STARTING STANDARD SB3 DQN (BASELINE)")
    print(f"{'='*30}")
    
    std_train_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=999)
    std_train_env = VecFrameStack(std_train_env, n_stack=4)
    std_train_env = VecTransposeImage(std_train_env)
    
    std_model = DQN(
        policy="CnnPolicy",
        env=std_train_env,
        verbose=0,
        device="auto",
        **DQN_HYPERPARAMS_STD
    )
    
    std_storage = {'times': [], 'updates': [], 'steps': [], 'rewards': []}
    
    callback = VanillaLoggingCallback(eval_env, EVAL_FREQ, std_storage)
    
    try:
        std_model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    except KeyboardInterrupt:
        print("Standard training interrupted.")
    
    std_model.save("dqn_pong_standard")
    all_results["Standard DQN"] = std_storage
    std_train_env.close()
    
    eval_env.close()
    print("\nВсе обучения завершены. Построение графиков...")
    plot_results(all_results)
    print("Готово.")

if __name__ == '__main__':
    train()
