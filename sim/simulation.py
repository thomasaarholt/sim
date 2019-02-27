def prismatic(file, limits, label="", PRISM=True, savepath=None, thermal_effects=True, total_FP=50, probestep=0.15, sliceThickness=1.6218179):
    import os
    import pyprismatic as pr
    from IPython.display import display
    import random
    from pathlib import Path
    file = os.path.abspath(file)
    
    _, filename = os.path.split(file)
    name, _ = os.path.splitext(filename)

    if savepath is None:
        savepath = 'prism' if PRISM else 'multislice'
    Path(savepath).mkdir(parents=True, exist_ok=True)

    no_thermal_label = "_therm" if thermal_effects else "_notherm"
    algorithm = 'prism' if PRISM else 'multislice'

    title = name + label + no_thermal_label
    print(title)
    existing_files = list(Path(savepath).glob(title + '*'))
    if existing_files:
        nums = []
        for l in existing_files:
            nums.append(int(l.stem.split("_FP")[1]))
        firstFP = max(nums) + 1
    else:
        firstFP = 0

    XMin, XMax = limits
    YMin, YMax = limits

    list_of_random_numbers = []
    for FP_number in range(firstFP,total_FP):
        output = os.path.join(savepath, name + label + no_thermal_label + '_FP{:03d}.mrc'.format(FP_number))

        probestep = 0.15 # Å
        potential_spacing = 0.05 # Å
        include_thermal_effects = thermal_effects
        alpha=20.0e-3
        focus=0

        random_number = random.randint(0,10000)
        list_of_random_numbers.append(str(random_number))

        meta = pr.Metadata(
            filenameAtoms=file, 
            algorithm=algorithm,
            E0=300e3,
            potBound = 2.0,
            probeSemiangle=alpha,
            alphaBeamMax = alpha+4e-3,
            interpolationFactorX=4, 
            interpolationFactorY=4, 
            filenameOutput=output,
            probeStepX = probestep,
            probeStepY = probestep,
            realspacePixelSizeX = potential_spacing,
            realspacePixelSizeY = potential_spacing,
            detectorAngleStep = 0.001,
            sliceThickness = sliceThickness,
            scanWindowXMin = XMin,
            scanWindowXMax = XMax,
            scanWindowYMin = YMin, 
            scanWindowYMax = YMax,
            numFP = 1,
            probeDefocus = focus,
            numGPUs=4,
            numThreads=20,
            includeThermalEffects=include_thermal_effects,
            tileX=1,
            tileY=1,
            tileZ=1,
            transferMode='singlexfer',
            randomSeed=random_number,
            numStreamsPerGPU=3,
            batchSizeTargetGPU=1,
        )
        
        from time import time
        t1 = time()
        meta.go()
        t2 = time()
        display('It took {:.2f} minutes, or {:.2f} hours'.format((t2 - t1)/60, (t2 - t1)/3600))

    random_filename = os.path.join(savepath, name + label + '_random.txt')
    with open(random_filename, 'a') as f:
        f.write("\n".join(list_of_random_numbers))
