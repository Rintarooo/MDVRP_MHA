#!/bin/bash

echo "run $0"
n_depot=3
n_car_each_depot=5
n_customer=20
seed=0
capa=1.

# echo "n_depot: ${n_depot}"
python dataclass.py ${n_depot} ${n_car_each_depot} \
	${n_customer} ${seed} ${capa}

 
