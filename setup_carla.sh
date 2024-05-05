#!/usr/bin/env bash

# Download and install CARLA
mkdir carla
cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.11.tar.gz
tar -xf CARLA_0.9.11.tar.gz
rm CARLA_0.9.11.tar.gz
cd ..
