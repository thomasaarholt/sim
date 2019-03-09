from sim.fileio import readMRC
import os
import pyprismatic as pr
from IPython.display import display
import random
from pathlib import Path
from time import time
import numpy as np


def make_file(output):
    try:
        print('Reading {}'.format(Path(output).stem + ".."), end='')
        data = readMRC(output)
    except:
        return True  # file does not exist, so continue making it
    if np.allclose(data[0, 0, 0], data[0, 0]):
        print('....identical data: Broken!')
        return True  # File is broken, remake file
    elif np.isnan(data).any():
        print('....file has nans: Broken!')
        return True  # File is broken, remake file
    elif np.isinf(data).any():
        print('....file has infs: Broken!')
        return True  # File is broken, remake file
    # elif data.min() < 1e-15:
    #     print('....data has really small values: Broken!')
    #     return True  # File is broken, remake file
    elif data.max() > 1:
        print('....data has more than one count total: Broken!')
        return True  # File is broken, remake file
    else:
        print('..ok!')
        return False  # File is fine


def prismatic(file, limits, label="", PRISM=True, savepath=None,
              thermal_effects=True, total_FP=50, probestep=0.15,
              sliceThickness=1.6218179, numGPUs=4, defocus_delta=0, tile=1,):

    file = os.path.abspath(file)

    _, filename = os.path.split(file)
    name, _ = os.path.splitext(filename)

    if savepath is None:
        savepath = 'prism' if PRISM else 'multislice'
    Path(savepath).mkdir(parents=True, exist_ok=True)

    no_thermal_label = "_therm" if thermal_effects else "_notherm"
    algorithm = 'prism' if PRISM else 'multislice'

    XMin, XMax = limits
    YMin, YMax = limits

    tileX = tileY = tile

    random_filename = os.path.join(savepath, name + label + '_random.txt')

    for FP_number in range(total_FP):
        output = os.path.join(
            savepath, name + label + no_thermal_label +
            '_FP{:03d}.mrc'.format(FP_number))

        probestep = 0.15  # Å
        potential_spacing = 0.05  # Å
        include_thermal_effects = thermal_effects
        alpha = 20.0e-3
        focus = 0

        random_number = random.randint(0, 100000)
        with open(random_filename, 'a') as f:
            f.write(str(random_number) + "\n")
        meta = pr.Metadata(
            filenameAtoms=file,
            algorithm=algorithm,
            E0=300e3,
            potBound=2.0,
            probeSemiangle=alpha,
            alphaBeamMax=alpha+2e-3,
            interpolationFactorX=4,
            interpolationFactorY=4,
            filenameOutput=output,
            probeStepX=probestep,
            probeStepY=probestep,
            realspacePixelSizeX=potential_spacing,
            realspacePixelSizeY=potential_spacing,
            detectorAngleStep=0.001,
            sliceThickness=sliceThickness,
            scanWindowXMin=XMin,
            scanWindowXMax=XMax,
            scanWindowYMin=YMin,
            scanWindowYMax=YMax,
            numFP=1,
            probeDefocus=focus + defocus_delta,
            numGPUs=numGPUs,
            alsoDoCPUWork=False,
            numThreads=1,
            includeThermalEffects=include_thermal_effects,
            tileX=tileX,
            tileY=tileY,
            tileZ=1,
            transferMode='singlexfer',
            randomSeed=random_number,
            numStreamsPerGPU=3,
            batchSizeTargetGPU=1,
        )
        while make_file(output):
            t1 = time()
            meta.go()
            t2 = time()
            display('It took {:.2f} minutes, or {:.2f} hours'.format(
                (t2 - t1)/60, (t2 - t1)/3600))
