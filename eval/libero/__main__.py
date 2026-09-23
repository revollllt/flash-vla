"""LIBERO task-success evaluation for OpenPI Pi0.5 and Flash-VLA.

Uses the episode protocol of Physical-Intelligence/openpi/examples/libero:
256px rendering, 180-degree image rotation, 10 settling steps, replan every 5.
Exceptions abort the run; they are not counted as policy failures.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}


def observation(obs: dict, prompt: str) -> dict:
    from PIL import Image

    def image(key):
        rotated = np.ascontiguousarray(obs[key][::-1, ::-1])
        return np.asarray(Image.fromarray(rotated).resize((224, 224), Image.Resampling.BILINEAR))

    quat = obs["robot0_eef_quat"]
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    axis_angle = np.zeros(3) if den == 0.0 else quat[:3] * (2.0 * np.arccos(w) / den)
    return dict(image=image("agentview_image"), wrist_image=image("robot0_eye_in_hand_image"),
                state=np.concatenate((obs["robot0_eef_pos"], axis_angle, obs["robot0_gripper_qpos"])),
                prompt=prompt)


def episode(env, policy, obs, *, prompt, max_steps, replan_steps, seed, fixture_path):
    rng = np.random.default_rng(seed)
    for _ in range(10):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    actions = deque()
    inference_seconds = 0.0
    calls = 0
    for step in range(max_steps):
        if not actions:
            inputs = observation(obs, prompt)
            noise = rng.standard_normal((10, 32), dtype=np.float32)
            if calls == 0 and fixture_path is not None:
                np.savez(fixture_path, **inputs, noise=noise)
            start = time.perf_counter()
            output = policy.infer(inputs, noise)
            inference_seconds += time.perf_counter() - start
            calls += 1
            actions.extend(output["actions"][:replan_steps])
        obs, _, done, _ = env.step(actions.popleft().tolist())
        if done:
            return dict(success=True, steps=step + 1, inference_calls=calls,
                        inference_seconds=inference_seconds)
    return dict(success=False, steps=max_steps, inference_calls=calls,
                inference_seconds=inference_seconds)


def evaluate(args, policy):
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = list(range(suite.n_tasks)) if args.task_ids is None else args.task_ids
    report = dict(policy=policy.metadata, suite=args.suite, seed=args.seed,
                  trials_per_task=args.trials, task_ids=task_ids, replan_steps=args.replan_steps,
                  source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_changes=subprocess.check_output(["git", "status", "--short"], text=True),
                  renderer=os.environ["MUJOCO_GL"],
                  episode_seed="seed + task_id * 1000 + trial",
                  noise="NumPy default_rng(episode_seed), float32 standard_normal rounded to BF16, chunk [10,32]",
                  status="running", episodes=[], completed=0, successes=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for task_id in task_ids:
        task = suite.get_task(task_id)
        # LIBERO stores trusted NumPy initial states in legacy torch pickle files.
        states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder /
                            task.init_states_file, weights_only=False)
        if args.trials > len(states):
            raise ValueError(f"Requested {args.trials} unique initial states; task {task_id} has {len(states)}")
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=256, camera_widths=256)
        try:
            for trial in range(args.trials):
                episode_seed = args.seed + task_id * 1000 + trial
                # A reset per episode prevents earlier trajectories from changing later RNG states.
                np.random.seed(episode_seed)
                env.seed(episode_seed)
                env.reset()
                obs = env.set_init_state(states[trial])
                fixture_path = args.out.with_suffix(".fixture.npz") if not report["episodes"] else None
                start = time.perf_counter()
                result = episode(env, policy, obs, prompt=task.language, max_steps=MAX_STEPS[args.suite],
                                 replan_steps=args.replan_steps, seed=episode_seed, fixture_path=fixture_path)
                result.update(task_id=task_id, trial=trial, seed=episode_seed,
                              prompt=task.language, elapsed_seconds=time.perf_counter() - start)
                report["episodes"].append(result)
                report["completed"] += 1
                report["successes"] += int(result["success"])
                report["success_rate"] = report["successes"] / report["completed"]
                args.out.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        finally:
            env.close()
    report["status"] = "complete"
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Success: {report['successes']}/{report['completed']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["flashvla", "official"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=os.environ.get("PALIGEMMA_TOKENIZER"))
    parser.add_argument("--target", default="rtx5090/pi05")
    parser.add_argument("--plan", default="shipped")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--suite", choices=MAX_STEPS, default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, help="Evaluate one saved observation without running the simulator")
    parser.add_argument("--exact-rope", action="store_true", help="Numerical diagnosis only: restore FP32 RoPE buffers in the official model")
    args = parser.parse_args()
    if args.trials < 1 or not 1 <= args.replan_steps <= 10:
        parser.error("trials must be positive; replan-steps must be between 1 and 10")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from .policy import Pi05LiberoPolicy
    import torch

    torch.set_num_threads(8)
    policy = Pi05LiberoPolicy(args.checkpoint, args.tokenizer, engine=args.engine,
                             target=args.target, plan=args.plan, steps=args.steps,
                             exact_rope=args.exact_rope)
    if args.fixture:
        data = dict(np.load(args.fixture))
        output = policy.infer(data, data["noise"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out.with_suffix(".npz"), **output)
        args.out.write_text(json.dumps(dict(policy=policy.metadata, fixture=str(args.fixture)), indent=2) + "\n")
    else:
        evaluate(args, policy)


if __name__ == "__main__":
    main()
