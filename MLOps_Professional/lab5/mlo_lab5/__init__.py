
from os import getenv, environ
from pathlib import Path
from sys import path

here = Path(__file__).parent
# Add parent directory to path
path.append(str(Path.cwd().parent))


dotenv_path = here / ".env"
if dotenv_path.exists():
    lines = [
        x.split("#")[0].strip()
        if "#" in x
        else x
        for x in
        dotenv_path.read_text().splitlines()
    ]
    pairs = [
        x.split("=")
        for x in lines
        if x
    ]
    for key, value in pairs:
        if key not in getenv():
            print(f"Setting {key} to {value}")
            getenv()[key] = value

print(getenv("OMP_NUM_THREADS"))
