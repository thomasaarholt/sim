from pathlib import Path
from shutil import copy
def copy_file(target_folder):
    # src = str(Path(__file__).parent / 'run.py')
    # dest = str(target_folder)
    # copy(src, dest)

    src = str(Path(__file__).parent / 'save.py')
    dest = str(target_folder)
    copy(src, dest)