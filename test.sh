#!/bin/bash

echo "running $0 ..."
n_depot=3
n_car_each_depot=5
n_customer=20
#seed=0
#capa is float value
capa=1

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
elif [ "$1" = "g1" ]; then
	seed=1
	python dataclass.py ${n_depot} ${n_car_each_depot} \
		${n_customer} ${seed} ${capa}
elif [ "$1" = "ga" ]; then
	write_csv="Csv/n${n_customer}d${n_depot}c${n_car_each_depot}D${capa}_ga.csv"
	if [ -e ${write_csv} ]; then
		rm -rf ${write_csv}
	fi

	for seed in $(seq 1 10)
	do
		filename=n${n_customer}d${n_depot}c${n_car_each_depot}D${capa}s${seed}
		if [ -e "GA/data/${filename}.txt" ]; then
			python GA/main.py GA/data/${filename}.txt ${write_csv}
		fi
	done
	python Csv/get_mean.py ${write_csv}
elif [ "$1" = "or" ]; then
	write_csv="Csv/n${n_customer}d${n_depot}c${n_car_each_depot}D${capa}_or.csv"
	if [ -e ${write_csv} ]; then
		rm -rf ${write_csv}
	fi

	for seed in $(seq 1 10)
	do
		filename=n${n_customer}d${n_depot}c${n_car_each_depot}D${capa}s${seed}
		if [ -e "Ortools/data/${filename}.json" ]; then
			python Ortools/solver.py -p Ortools/data/${filename}.json -c ${write_csv}
		fi
	done
	python Csv/get_mean.py ${write_csv}
elif [ "$1" = "to" ]; then
	write_csv="Csv/n${n_customer}d${n_depot}c${n_car_each_depot}D${capa}_to.csv"
	write_csv_2opt="Csv/n${n_customer}d${n_depot}c${n_car_each_depot}D${capa}_to_2opt.csv"
	if [ -e ${write_csv} ]; then
		rm -rf ${write_csv}
		rm -rf ${write_csv_2opt}
	fi

	for seed in $(seq 1 10)
	do
		filename=n${n_customer}d${n_depot}c${n_car_each_depot}D${capa}s${seed}
		if [ -e "Torch/data/${filename}.json" ]; then
			python Torch/plot.py -p Torch/Weights/VRP20_epoch23.pt -t Torch/data/${filename}.json -wc ${write_csv} -wc2 ${write_csv_2opt} -b 512
		fi
	done
	python Csv/torch_mean.py ${write_csv}
	python Csv/torch_mean.py ${write_csv_2opt}
else
	echo "command: ${0} g or ${0} rm"
fi