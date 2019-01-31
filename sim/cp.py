from pathlib import Path
from shutil import copy
def copy_file(target_folder):
    copy(Path(__file__).parent / 'run.py', target_folder)