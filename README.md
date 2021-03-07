# MDVRP solver
implementation of MDVRP solver
* Deep Reinforcement Learning(Policy Gradient, model architecture has Multi-Head Attention layer)
* GA(Genetic Algorithm)
* Google OR-Tools(https://developers.google.com/optimization/routing)

![Screen Shot 2021-02-20 at 4 24 02 PM](https://user-images.githubusercontent.com/51239551/108587625-1b967300-7398-11eb-9c45-a3af10bdb343.png)


![newplot](https://user-images.githubusercontent.com/51239551/104798863-88ed3c00-580d-11eb-852c-09c88f2f9afc.png)

```bash

├── Csv -> mean of cost and time during test 
├── Png -> plot images during test
│
├── GA
│   └── data
│
├── Ortools
│   └── data
│
└── Torch
    ├── data
    ├── Nets -> python codes for neural network
    ├── Pkl -> pickle files contaning hyperparameter
    ├── Weights -> pt files of pre-trained weights 
    └── Csv -> csv files of train log
```

## Environment
I leave my own environment below. I tested it out on a single GPU.
* OS:
	* Linux(Ubuntu 18.04.5 LTS) 
* GPU:
	* NVIDIA® GeForce® RTX 2080 Ti VENTUS 11GB OC
* CPU:
	* Intel® Xeon® CPU E5640 @ 2.67GHz
* NVIDIA® Driver = 455.45.01
* Docker = 20.10.3
* [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker)(for GPU)

### Dependencies

* Python = 3.6.10
* PyTorch = 1.6.0
* scipy
* numpy
* plotly(only for plotting)
* matplotlib(only for plotting in GA code)
* pandas(only for mean of test score)

### Docker(option)
Make sure you've already installed `Docker`
```bash
docker version
```
latest `NVIDIA® Driver`
```bash
nvidia-smi
```
and `nvidia-docker2`(for GPU)
<br>
#### Usage

1. build or pull docker image

build image
```bash
./docker.sh build
```
pull image from [dockerhub](https://hub.docker.com/repository/docker/docker4rintarooo/mdvrp/tags?page=1&ordering=last_updated)
```bash
docker pull docker4rintarooo/mdvrp:latest
```

2. run container using docker image(-v option is to mount directory)
```bash
./docker.sh run
```
If you don't have a GPU, you can run
```bash
./docker.sh run_cpu
```
<br><br>


## Usage
* train
* inference(with 10 data)
* inference(with 1 data)
<br><br>

### train phase

First move to `Torch` dir. 

```
cd Torch
```

Then, generate the pickle file contaning hyperparameter values by running the following command.

```
python config.py
```

you would see the pickle file in `Pkl` dir. now you can start training the model.

```
python train.py -p Pkl/***.pkl
```  
<br><br>

### inference phase(with 10 test data using shell script)

set parameter(n_depot, n_car_each_depot, n_customer, capa) by editing `test.sh` and run it.
<br>
`g` option generates 10 test data in `Torch/data`, `Ortools/data` and `GA/data` dir.
```
./test.sh g
```
If you create data by mistake, you can remove them with `rm` option
```
./test.sh rm
```

Now you can test with `or`, `to` and `ga` option.  
You can see the result score in `Csv` dir.
<br><br>

### inference phase(with 1 test data manually)
Generate test data
  
(GA -> txt file, Torch and Ortools -> json file).

```
python dataclass.py
```

Plot prediction of the pretrained model
```
cd Torch && python plot.py -p Weights/***.pt -t data/***.json -b 128
```
Compare the results 
```
cd GA && python main.py data/***.txt
```
```
cd Ortools && python main.py -p data/***.json
```

## Reference
### DRL_MHA
* https://github.com/Rintarooo/VRP_DRL_MHA
* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/d-eremeev/ADM-VRP
* https://qiita.com/ohtaman/items/0c383da89516d03c3ac0 (written in Japanese)

### Google Or-Tools
* https://github.com/skatsuta/vrp-solver

### GA(Python)
* https://github.com/Lagostra/MDVRP

### GA(C++)
* https://github.com/mathiasaap/Multi-Depot-Routing---Genetic-algorithm

### GA(public data)
* https://neo.lcc.uma.es/vrp/vrp-instances/multiple-depot-vrp-instances/