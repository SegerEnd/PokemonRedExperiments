import sys
from os.path import exists
from pathlib import Path
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
import torch
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
from best_agent_viewer import BestAgentViewer

USER = "Linux"
COLOR = "#87CEFA"

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = StreamWrapper(
            RedGymEnv(env_conf),
            stream_metadata = { # All of this is part is optional
                "user": USER, # choose your own username
                "env_id": rank, # environment identifier
                "color": COLOR, # choose your color :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    use_wandb_logging = False
    ep_length = 2048 * 80
    sess_id = "runs"
    sess_path = Path(sess_id)

    env_config = {
                'headless': True, 'save_final_state': False, 'early_stop': False,
                'action_freq': 24, 'init_state': '../init.state', 'max_steps': ep_length,
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'reward_scale': 0.5, 'explore_weight': 0.15
            }

    print(env_config)

    num_cpu = 64 # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    checkpoint_callback = CheckpointCallback(save_freq=ep_length//2, save_path=sess_path,
                                     name_prefix="poke")

    callbacks = [checkpoint_callback, TensorboardCallback(sess_path), BestAgentViewer(user=USER, color=COLOR)]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            name="v2-a",
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    #env_checker.check_env(env)

    # put a checkpoint here you want to start from
    if sys.stdin.isatty():
        file_name = ""
    else:
        file_name = sys.stdin.read().strip() #"runs/poke_26214400_steps"

    train_steps_batch = ep_length // 64

    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        if exists(file_name + "_vecnorm.pkl"):
            env = VecNormalize.load(file_name + "_vecnorm.pkl", env.venv)
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "MultiInputPolicy", env, verbose=1,
            n_steps=train_steps_batch, batch_size=2048, n_epochs=1,
            gamma=0.998, gae_lambda=0.95, ent_coef=0.02,
            learning_rate=2.5e-4, tensorboard_log=sess_path,
            device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        )

    print(model.policy)

    try:
        model.learn(total_timesteps=(ep_length)*num_cpu*10000, callback=CallbackList(callbacks), tb_log_name="poke_ppo", reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        model.save(sess_path / "poke_latest")
        env.save(sess_path / "poke_latest_vecnorm.pkl")
        print(f"Model saved to {sess_path / 'poke_latest.zip'}")

    if use_wandb_logging:
        run.finish()
