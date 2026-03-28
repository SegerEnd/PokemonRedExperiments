import os
from pathlib import Path
import uuid

from red_gym_env_v2 import RedGymEnv
from baseline_sample_factory import make_pokemon_env, register_pokemon_env

from sample_factory.cfg.arguments import parse_sf_args
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.learning.learner import Learner
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.utils import log
from sample_factory.algo.utils.context import sf_global_context

import torch


def load_sf_model(train_dir, experiment):
    """Load a Sample Factory checkpoint and return the actor-critic model."""
    register_pokemon_env()

    parser, cfg = parse_sf_args(argv=[
        f"--env=pokemon_red",
        f"--train_dir={train_dir}",
        f"--experiment={experiment}",
    ])

    cfg.num_workers = 1
    cfg.num_envs_per_worker = 1

    # Find latest checkpoint
    checkpoint_dir = Path(train_dir) / experiment
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"), key=os.path.getmtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest = checkpoints[-1]
    checkpoint_file = latest / "checkpoint.pth"
    print(f"Loading checkpoint: {latest.name}")

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    # Create env to get observation/action spaces
    env_config = AttrDict(worker_index=0, vector_index=0)
    env = make_pokemon_env("pokemon_red", cfg=cfg, env_config=env_config)
    obs_space = env.observation_space
    action_space = env.action_space
    env.close()

    # Build model and load weights
    model = create_actor_critic(cfg, obs_space, action_space)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


if __name__ == "__main__":
    sess_path = Path(f"session_{str(uuid.uuid4())[:8]}")
    ep_length = 2**23

    env_config = {
        "headless": False,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "../init.state",
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": "../PokemonRed.gb",
        "debug": False,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
    }

    env = RedGymEnv(env_config)
    model = load_sf_model(train_dir="./runs_sf", experiment="poke_sf")

    obs, info = env.reset()
    rnn_states = None

    while True:
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except Exception:
            agent_enabled = False

        if agent_enabled:
            # Convert obs dict to tensors
            obs_tensors = {
                k: torch.from_numpy(v).unsqueeze(0).float()
                for k, v in obs.items()
            }
            with torch.no_grad():
                policy_output = model(obs_tensors, rnn_states)
                action = policy_output["actions"].squeeze().item()
                rnn_states = policy_output.get("new_rnn_states", None)

            obs, rewards, terminated, truncated, info = env.step(action)
        else:
            env.pyboy.tick(1, True)
            obs = env._get_obs()
            truncated = env.step_count >= env.max_steps - 1

        env.render()
        if truncated:
            break

    env.close()
