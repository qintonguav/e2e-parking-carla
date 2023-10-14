#!/usr/bin/env bash

# Download and install CARLA
mkdir carla
cd carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz
tar -xf CARLA_0.9.11.tar.gz
rm CARLA_0.9.11.tar.gz
cd ..