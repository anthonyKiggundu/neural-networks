import subprocess

CONFIGS = [
    "configs/base.yaml",
    "configs/long_rollout.yaml",
    "configs/high_capacity.yaml",
]

for cfg in CONFIGS:
    print("Running", cfg)
    subprocess.run(["python", "run_experiment.py", cfg])

