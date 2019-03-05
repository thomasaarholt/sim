from sim import stack_and_save
from sim import prismatic
from sim.defocus import get_series_defocus_and_weight
import pathlib
files = []
for p in pathlib.Path('./Models').iterdir():
    if "_Prismatic" in p.name:
        files.append(str(p))

# def key(entry):
#     return int(entry.split('_d')[1].split('_')[0])

files = sorted(files)  # , key=key)

sigma, defocus_list, weighting_list = get_series_defocus_and_weight(
    ZLP=0.9, Voltage=3e5, Chromatic=1.6e-3)

for filename in files:
    for i, defocus_delta in defocus_list:
        prismatic(
            filename, label="defocus_"+str(i), limits=({0}, {1}), 
            PRISM=True, savepath='prism', thermal_effects=True, 
            total_FP=50, sliceThickness=1.6218179, defocus_delta=defocus_delta)

stack_and_save()
