from sim import prismatic
import pathlib
files = []
for p in pathlib.Path('./Models').iterdir():
    if "_Prismatic" in p.name:
        files.append(str(p))

# def key(entry):
#     return int(entry.split('_d')[1].split('_')[0])

files = sorted(files)#, key=key)

for filename in files:
    prismatic(filename, label="_slice1.6", limits = ({0}, {1}), PRISM=True, savepath='prism', thermal_effects=False, total_FP=1, sliceThickness=1.6218179)
    prismatic(filename, label="_slice1.6", limits = ({0}, {1}), PRISM=True, savepath='prism', thermal_effects=True, total_FP=20, sliceThickness=1.6218179)

from sim import stack_and_save
stack_and_save()
