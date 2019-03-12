```
# In .profile
export CPLUS_INCLUDE_PATH=/opt/pgi/linux86-64/2018/cuda/10.0/include:$SHARED/compiled/boost:$SHARED/compiled/fftw/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$SHARED/compiled/fftw/lib:/opt/pgi/linux86-64/2018/cuda/10.0/lib64:$SHARED/compiled/prismatic/lib$LD_LIBRARY_PATH
export LIBRARY_PATH=$SHARED/compiled/fftw/lib:/opt/pgi/linux86-64/2018/cuda/10.0/lib64:$SHARED/compiled/prismatic/lib:$LIBRARY_PATH

# Commands
pip install --install-option="--enable-gpu" --user https://github.com/prism-em/prismatic/archive/master.zip
pip install --user atomap ase
pip install --user https://github.com/thomasaarholt/sim/archive/master.zip

ssh-copy-id -i ~/.ssh/mykey thomasaar@ml1.hpc.uio.no
```
