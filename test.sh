#!/bin/bash

echo "running $0 ..."
n_depot=3
n_car_each_depot=5
n_customer=20
#seed=0
capa=1
#capa is float value

# echo "n_depot: ${n_depot}"
if [ "$1" = "g" ]; then
	for seed in $(seq 1 10)
	do
		python dataclass.py ${n_depot} ${n_car_each_depot} \
			${n_customer} ${seed} ${capa}
	done
elif [ "$1" = "rm" ]; then
	for seed in $(seq 1 10)
	do
		echo "rm"
		filename=n${n_customer}d${n_depot}c${n_car_each_depot}D${capa}s${seed}
		rm -rf Torch/data/${filename}.json
		rm -rf Ortools/data/${filename}.json
		rm -rf GA/data/${filename}.txt
	done
else
	echo "command: ${0} g or ${0} rm"
fi