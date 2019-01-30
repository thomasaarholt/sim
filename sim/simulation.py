def prismatic(file, limits, PRISM=True, savepath=None, thermal_effects=True, firstFP=0, total_FP=50):
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

    no_thermal_label = "" if thermal_effects else "_nothermal"
    algorithm = 'prism' if PRISM else 'multislice'
    XMin, XMax = limits
    YMin, YMax = limits

    for FP_number in range(firstFP,total_FP):
        output = os.path.join(savepath, name + '_FP{:03d}{}.mrc'.format(FP_number, no_thermal_label))

        probestep = 0.15 # Å
        potential_spacing = 0.025 # Å
        include_thermal_effects = thermal_effects
        alpha=20.0e-3
        focus=0

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
            sliceThickness = 1.6218179,
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
            randomSeed=random.randint(0,10000),
            numStreamsPerGPU=3,
            batchSizeTargetGPU=1,
        )
        
        from time import time
        t1 = time()
        meta.go()
        t2 = time()
        display('It took {:.2f} minutes, or {:.2f} hours'.format((t2 - t1)/60, (t2 - t1)/3600))