from os import getenv
from alexlib.files.config import DotEnv

# Load environment variables from .env file
envs = DotEnv.from_start(__file__)

print(getenv("OMP_NUM_THREADS"))