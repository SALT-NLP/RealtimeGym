import argparse
import importlib
import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from types import ModuleType
from typing import Any

import pandas as pd
import pygame
import yaml
from PIL import Image

import realtimegym
from realtimegym.agents.agile import AgileThinker
from realtimegym.agents.planning import PlanningAgent
from realtimegym.agents.reactive import ReactiveAgent


def _load_prompt_module(specifier: str) -> ModuleType:
    """
    Load prompt module from:
      - dotted module name (recommended), e.g. "realtimegym.prompts.freeway"
      - file path (backward compatibility), e.g. "configs/prompts/freeway.py"
      - path-like with slashes, e.g. "configs/prompts/freeway"
    Returns imported module.
    """
    project_root = os.getcwd()

    # 1) Try resolve as file path (with or without .py)
    candidates = [specifier]
    if not specifier.endswith(".py"):
        candidates.append(specifier + ".py")
    # also try relative to project root
    candidates += [os.path.join(project_root, c) for c in list(candidates)]
    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            name = "realtimegym._prompt_" + os.path.splitext(os.path.basename(p))[0]
            spec = importlib.util.spec_from_file_location(name, p)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            return module

    # 2) Try convert slashes to dots and import as module
    modname = specifier.replace("/", ".")
    if modname.endswith(".py"):
        modname = modname[:-3]
    # ensure project root on sys.path so 'configs' can be found if it's not a package
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        return importlib.import_module(modname)
    except Exception as e:
        raise ImportError(
            f"Cannot load prompt module '{specifier}' as path or module ({e}). "
            "Recommended: use module name (e.g. realtimegym.prompts.freeway). "
            "For backward compatibility, file paths are also supported (e.g. configs/prompts/freeway.py)."
        )


def check_args(args: argparse.Namespace) -> None:
    if args.mode == "planning":
        assert args.internal_budget == 0, (
            "Internal budget must be 0 when mode is planning."
        )
    else:
        assert args.internal_budget > 0, (
            "Internal budget must be greater than 0 when mode is reactive or agile."
        )
    assert args.internal_budget <= args.time_pressure, (
        "Internal budget must be less than or equal to time pressure when method is fast."
    )


def game_loop(file: str, raw_seed: int, args: argparse.Namespace) -> dict[str, Any]:
    version = (
        "0"
        if args.cognitive_load == "E"
        else "1"
        if args.cognitive_load == "M"
        else "2"
    )

    env, seed, render = realtimegym.make(
        f"{args.game.capitalize()}-v{version}",
        seed=raw_seed,
        render=args.save_trajectory_gifs,
    )
    with open(args.prompt_config, "r") as f:
        prompt_config = yaml.safe_load(f)
    assert args.game in prompt_config, "Game not found in prompt config."

    params = {
        "prompts": _load_prompt_module(prompt_config[args.game]),
        "file": file,
        "time_unit": args.time_unit,
    }

    if args.mode != "reactive":
        params["model2_config"] = args.planning_model_config

    if args.mode != "planning":
        params["model1_config"] = args.reactive_model_config
        params["internal_budget"] = args.internal_budget

    if args.mode == "planning":
        params["skip_action"] = True
        if args.game == "overcooked":
            params["skip_action"] = False

    if args.mode == "reactive":
        agent = ReactiveAgent(**params)  # type: ignore
    elif args.mode == "planning":
        agent = PlanningAgent(**params)  # type: ignore
    elif args.mode == "agile":
        agent = AgileThinker(**params)  # type: ignore
        agent.configure_refresh(
            refresh_plan=args.refresh_plan,
            refresh_every=args.refresh_every,
            refresh_on_change=args.refresh_on_change,
            refresh_change_threshold=args.refresh_change_threshold,
            refresh_carryover=args.refresh_carryover,
        )
    else:
        raise NotImplementedError("mode not recognized.")

    if args.checkpoint is not None:  # resume from checkpoint
        checkpoint_file = file.replace(args.log_dir, args.checkpoint)
        df = pd.read_csv(checkpoint_file)
        obs, done = env.reset()
        for a in df["action"]:
            obs, done, reward, reset_flag = env.step(a)
        if done:
            ret = {
                "seed": seed,
                "reward": env.reward,
                "total_time": 0,
                "log_dir": os.path.dirname(file),
            }
            return ret
        else:
            env, seed, render = realtimegym.make(
                f"{args.game.capitalize()}-v{version}",
                seed=raw_seed,
                render=args.save_trajectory_gifs,
            )
            agent.resume_from_checkpoint(env, checkpoint_file)
    start_time = time.time()
    surfaces = []

    # New API loop: obs, done = env.reset()
    obs, done = env.reset()
    if render is not None:
        surfaces.append(render.render(env))
    steps = 0
    max_steps = getattr(args, "max_steps", None)
    while not done and (max_steps is None or steps < max_steps):
        agent.observe(obs)
        agent.think(timeout=args.time_pressure)
        action = agent.act()
        obs, done, reward, reset_flag = env.step(action)
        env.summary()
        agent.log(reward, reset_flag)
        steps += 1
        if render is not None:
            surfaces.append(render.render(env))
    if render is not None:
        gif_path = file.replace(".csv", ".gif")
        images = [pygame.surfarray.array3d(surface) for surface in surfaces]
        pil_images = [
            Image.fromarray(images[i].swapaxes(0, 1)) for i in range(len(images))
        ]
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=1000,
            loop=0,
        )
    ret = {
        "seed": seed,
        "reward": env.reward,
        "total_time": time.time() - start_time,
        "log_dir": os.path.dirname(file),
    }
    del env
    return ret


def main() -> None:
    """Main CLI entry point for the Real-time Reasoning Gym."""
    args = argparse.ArgumentParser(
        description="Real-time reasoning gym configurations."
    )
    args.add_argument("--planning-model-config", type=str, default=None)
    args.add_argument("--reactive-model-config", type=str, default=None)
    args.add_argument(
        "--prompt-config", type=str, default="configs/example-prompts.yaml"
    )
    args.add_argument(
        "--game",
        type=str,
        choices=["freeway", "snake", "overcooked"],
        default="freeway",
    )
    args.add_argument(
        "--time_unit", type=str, choices=["token", "seconds"], default="token"
    )
    args.add_argument("--time_pressure", type=int, default=8192)
    args.add_argument("--cognitive_load", type=str, choices=["E", "M", "H"])
    args.add_argument("--mode", type=str, choices=["agile", "reactive", "planning"])
    args.add_argument("--internal_budget", type=int, default=8192)
    args.add_argument("--max_steps", type=int, default=None,
                      help="cap env steps per episode (default: run to natural termination)")
    # Gap 3 — planning-thread refresh (agile mode). Off by default = vanilla.
    args.add_argument("--refresh_plan", action="store_true", default=False)
    args.add_argument("--refresh_every", type=int, default=0)
    args.add_argument("--refresh_on_change", action="store_true", default=False)
    args.add_argument("--refresh_change_threshold", type=float, default=0.3)
    args.add_argument("--refresh_carryover", action="store_true", default=False)
    args.add_argument("--log_dir", type=str, default="logs")
    args.add_argument("--seed_num", type=int, default=1)
    args.add_argument("--repeat_times", type=int, default=1)
    args.add_argument("--save_trajectory_gifs", action="store_true", default=False)
    args.add_argument("--settings", type=str, nargs="+", default=[])
    args.add_argument("--instance_num", type=int, default=None)
    args.add_argument("--checkpoint", type=str, default=None)
    args = args.parse_args()
    if args.settings == []:
        args.settings = [
            f"{args.game}_{args.cognitive_load}_{args.time_pressure}_{args.mode}_{args.internal_budget}"
        ]
    instance = []
    for setting in args.settings:
        new_args = argparse.Namespace(**vars(args))
        game, cognitive_load, time_pressure, mode, internal_budget = setting.split("_")
        new_args.game = game
        new_args.cognitive_load = cognitive_load
        new_args.time_pressure = int(time_pressure)
        new_args.mode = mode
        new_args.internal_budget = int(internal_budget)
        check_args(new_args)
        log_dir = (
            new_args.log_dir + f"/{setting}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        for seed in range(args.seed_num):
            for r in range(args.repeat_times):
                if not os.path.exists(log_dir + f"/{r}_{seed}.csv"):
                    instance.append((log_dir + f"/{r}_{seed}.csv", seed, new_args))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(log_dir + "/args.log"):
            with open(log_dir + "/args.log", "w") as f:
                for arg, value in vars(new_args).items():
                    f.write(f"{arg}: {value}\n")
                f.write("\n")
    if args.instance_num is not None:
        assert args.instance_num == len(instance), "instance_num incorrect!"
    with ProcessPoolExecutor(max_workers=len(instance)) as executor:
        futures = [
            executor.submit(game_loop, log_file, seed, args)
            for (log_file, seed, args) in instance
        ]
        results = []
        total = len(futures)
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            log_dir = result["log_dir"]
            # remove log_dir from result
            del result["log_dir"]
            results.append(result)
            with open(f"{log_dir}/args.log", "a") as f:
                for key, value in result.items():
                    f.write(f"{key}: {value} ")
                f.write("\n-----------------------------\n")
            print(f"Progress: {idx}/{total} ({idx / total * 100:.2f}%)")


if __name__ == "__main__":
    main()
