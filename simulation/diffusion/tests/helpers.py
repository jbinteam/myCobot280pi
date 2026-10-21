"""Small deterministic demonstrations created once per test process."""
from functools import lru_cache
from pathlib import Path
import tempfile
from simulation.diffusion.generate_dataset import generate
from simulation.diffusion.planner import RRTConfig

_temporary_directories=[]


@lru_cache(maxsize=1)
def demonstrations():
    temporary=tempfile.TemporaryDirectory(prefix='mycobot-diffusion-tests-')
    _temporary_directories.append(temporary)
    path=Path(temporary.name)/'demonstrations.h5'
    generate(path,episodes=12,seed=42,config=RRTConfig(iterations=600,post_solution=20))
    return path
