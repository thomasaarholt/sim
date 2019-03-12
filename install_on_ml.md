# Installation instructions for building or installing pyprismatic
### Environment settings in Linux
**For version with GPU:**
In `~/.profile`, place the following:
```
export CPLUS_INCLUDE_PATH=/opt/pgi/linux86-64/2018/cuda/10.0/include:$SHARED/compiled/boost:$SHARED/compiled/fftw/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$SHARED/compiled/fftw/lib:/opt/pgi/linux86-64/2018/cuda/10.0/lib64:$SHARED/compiled/prismatic/lib$LD_LIBRARY_PATH
export LIBRARY_PATH=$SHARED/compiled/fftw/lib:/opt/pgi/linux86-64/2018/cuda/10.0/lib64:$SHARED/compiled/prismatic/lib:$LIBRARY_PATH
```
Then
```python
pip install --install-option="--enable-gpu" --user https://github.com/prism-em/prismatic/archive/master.zip # or from folder with .
pip install --user atomap ase
pip install --user https://github.com/thomasaarholt/sim/archive/master.zip
```

### Environment settings on Windows
**For version without GPU:**
You need some sort of compiler. You do *not* need Visual Studio, but do need to download and install VS Build Tools: https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017. Then set the following Environment Variables:
```
INCLUDE: C:\Users\thomasaar\Downloads\fftw3;C:\Users\thomasaar\Downloads\boost_1_69_0;
LIB: C:\Users\thomasaar\Downloads\fftw3;
```
Then:
```python
pip install https://github.com/prism-em/prismatic/archive/master.zip
```

### Other commands
ssh-copy-id -i ~/.ssh/mykey thomasaar@ml1.hpc.uio.no
```
