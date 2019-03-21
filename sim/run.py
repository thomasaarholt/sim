from sim.stack import stack_and_save
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
    for i, defocus_delta in enumerate(defocus_list):
        prismatic(
            filename, label="_defocus_"+str(i), limits=({0}, {1}),
            PRISM=True, savepath='prism', thermal_effects=True,
            total_FP=100, sliceThickness=0.8109, probestep=0.08,
            C3=10000, defocus_delta=defocus_delta)

        # prismatic(
        #     filename, label="_defocus_"+str(i), limits=({0}, {1}), 
        #     PRISM=True, savepath='prism', thermal_effects=False, 
        #     total_FP=1, sliceThickness=1.6218179, defocus_delta=defocus_delta)

stack_and_save(add_atom_positions=False)
