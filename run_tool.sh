#!/bin/bash

# Start up a MongoDB daemon instance
mkdir ./mongodb
/usr/bin/mongod --fork --logpath ./mongodb/log.txt --dbpath ./mongodb # have a local folder to drop everything
python gener_experiment.py --experiment experiments/cifar10 --optimize error --epochs 2 --constraint power --constraint_val 1000.0
