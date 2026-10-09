"""Run behavior contracts through the standard unittest suite."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if __name__ == "__main__":
    env = dict(os.environ, STUDIO_START_WORKERS="0")
    env["STUDIO_ROOT"] = tempfile.mkdtemp(prefix="modellab-suite-")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(Path(__file__).parent), "-p", "test_*.py"], check=True, env=env)
    print("CONTRATOS_COLAB_OK")
