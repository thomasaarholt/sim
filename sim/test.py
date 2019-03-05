def test():
    from pyprismatic.params import Metadata
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with open('temp.XYZ', 'w') as fid:
        fid.write("one unit cell of 100 silicon\n\
    5.43    5.43    5.43\n\
    14  0.0000  0.0000  0.0000  1.0  0.076\n\
    14  2.7150  2.7150  0.0000  1.0  0.076\n\
    14  1.3575  4.0725  1.3575  1.0  0.076\n\
    14  4.0725  1.3575  1.3575  1.0  0.076\n\
    14  2.7150  0.0000  2.7150  1.0  0.076\n\
    14  0.0000  2.7150  2.7150  1.0  0.076\n\
    14  1.3575  1.3575  4.0725  1.0  0.076\n\
    14  4.0725  4.0725  4.0725  1.0  0.076\n\
    -1")
    meta = Metadata(filenameAtoms='temp.XYZ', filenameOutput='output.mrc')
    meta.algorithm = 'multislice'
    meta.alsoDoCPUWork = False
    meta.go()
    os.remove("temp.XYZ")
    os.remove("output.mrc")