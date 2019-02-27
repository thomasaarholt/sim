pip install --install-option="--enable-gpu" --user https://github.com/prism-em/prismatic/archive/master.zip
pip install --user atomap ase
pip install --user https://github.com/thomasaarholt/sim/archive/master.zip

ssh-copy-id -i ~/.ssh/mykey thomasaar@ml1.hpc.uio.no
