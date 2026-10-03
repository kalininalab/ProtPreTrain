"""Check that a condor job sees the mounted repo, the GPU, the locked environment and the network.

Run by hpc/runs/smoke.txt through hpc/gpu.sub. A file rather than `python -c` because condor's argument parser
rejects unescaped double quotes.
"""

import sys
import urllib.request

import torch

import step

print("step      ", step.__file__)
print("python    ", sys.version.split()[0])
print("torch     ", torch.__version__, "built against CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible - check request_GPUs and the image")
print("device    ", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
print("arch list ", torch.cuda.get_arch_list())
x = torch.randn(1024, 1024, device="cuda")
print("matmul    ", float((x @ x).sum()))

from torch_geometric.nn import radius_graph  # noqa: E402

pos = torch.rand(200, 3, device="cuda") * 20
print("radius_graph on GPU:", tuple(radius_graph(pos, 5.0).shape))

import foldcomp  # noqa: E402,F401
import lightning  # noqa: E402
import mlflow  # noqa: E402

print("lightning ", lightning.__version__, "| mlflow", mlflow.__version__)

# Pretraining databases are downloaded inside jobs, so the execute node needs outbound HTTPS
with urllib.request.urlopen("https://foldcomp.steineggerlab.workers.dev/m_jannaschii.dbtype", timeout=30) as r:
    print("network    foldcomp server reachable, HTTP", r.status)
