#!/bin/bash
# python plot.py -p Weights/VRP100_epoch17.pt -t data/n20d3c5D1s2.json -b 3 -wc samp.csv -wc2 samp_2opt.csv
# python plot.py -p Weights/VRP20_epoch19.pt -t data/n20d3c5D1s2.json -b 512
# python plot.py -p Weights/VRP50_epoch18.pt -t data/n50d3c5D1s3.json -b 512
# python plot.py -p Weights/VRP100_epoch15.pt -t data/n100d3c2D3s2.json -b 512
# python plot.py -p Weights/VRP50_epoch18.pt -t data/n50d3c3D2s2.json -b 512
python plot.py -p Weights/VRP100_epoch15.pt -t data/n120d3c3D3s2.json -b 512
