import shutil
from argparse import ArgumentParser, ArgumentTypeError
from itertools import product
from pathlib import Path

sample_string = """
#!/bin/bash -l

#SBATCH --account=hai_preprost
#SBATCH --job-name=test
#SBATCH --partition=booster
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task={num_workers}
#SBATCH --mem=0
#SBATCH --time=0-24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
jutil env activate -p hai_preprost

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=ib3,ib2,ib1,ib0
export WANDB_DIR=$SCRATCH/wandb

conda activate step
srun python train.py""".strip()


def str_to_bool(value: str) -> bool:
    """Command line inputs that are bools."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def get_name(perm_dict: dict):
    """Concat keys and valus into a single string"""
    return "_".join([f"{k}_{v}" for k, v in perm_dict.items()])


def generate_config_permutations(input_dict: dict) -> tuple[list[dict], list[str]]:
    """For all config entries that are lists, generate a new config with all ."""
    fixed_items = {k: v for k, v in input_dict.items() if not isinstance(v, list)}
    variable_items = {k: v for k, v in input_dict.items() if isinstance(v, list)}
    if variable_items:
        keys, values = zip(*variable_items.items())
        permutations = [dict(zip(keys, v)) for v in product(*values)]
        config_list = [({**fixed_items, **perm}, get_name(perm)) for perm in permutations]
    else:
        config_list = [(fixed_items, "default")]
    return config_list


def config_to_str(d: dict) -> str:
    """From config generate argparse cli arguments"""
    res = ""
    for k, v in d.items():
        res += f" --{k} {v}"
    return res


parser = ArgumentParser()
# Arguments that shouldn't change between runs
parser.add_argument("--num_nodes", type=int, default=1, help="SLURM nodes")
parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epochs", type=int, default=5)
# Arguments to change across runs
parser.add_argument("--masktype", type=str, nargs="+", default="normal", choices=["normal", "ankh", "bert"])
parser.add_argument("--maskfrac", type=float, default=0.15, nargs="+")
parser.add_argument("--predict_all", type=str_to_bool, nargs="+", default=False)
parser.add_argument("--walk_length", type=int, nargs="+", default=20)
parser.add_argument("--alpha", type=float, nargs="+", default=0.5)
parser.add_argument("--posnoise", type=float, default=1.0, nargs="+")
args = parser.parse_args()

config = vars(args)
all_configs = generate_config_permutations(config)


p = Path("tmp_scripts/")
if p.exists():
    shutil.rmtree(p)
p.mkdir()
for config, filename in all_configs:
    with open(p / f"{filename}.sh", "w") as f:
        f.write(sample_string.format(num_nodes=config["num_nodes"], num_workers=config["num_workers"]))
        f.write(config_to_str(config))
