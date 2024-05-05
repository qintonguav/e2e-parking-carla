#!/usr/bin/env bash

# Download and install CARLA
mkdir carla
cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.11.tar.gz
tar -xf CARLA_0.9.11.tar.gz
rm CARLA_0.9.11.tar.gz

cd PythonAPI/carla/dist/
unzip carla-0.9.11-py3.7-linux-x86_64.egg -d carla-0.9.11-py3.7-linux-x86_64
cd carla-0.9.11-py3.7-linux-x86_64/
echo "install_carla"

cat>setup.py<<EOF
from distutils.core import setup
setup(name='carla', version='0.9.11', py_modules=['carla'],)
EOF

cd ..
pip install -e carla-0.9.11-py3.7-linux-x86_64
