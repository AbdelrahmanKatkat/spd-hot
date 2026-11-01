import subprocess
import sys


def test_package_import_does_not_load_training_frameworks():
    code = (
        "import sys, spd_hot; "
        "assert 'torch' not in sys.modules; "
        "assert 'ultralytics' not in sys.modules; "
        "assert 'spd_hot.train' not in sys.modules; "
        "assert 'shapely' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
