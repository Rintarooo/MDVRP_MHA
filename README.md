# MDVRP solver with Multi-Head Attention, GA, OR-Tools

![newplot](https://user-images.githubusercontent.com/51239551/104798863-88ed3c00-580d-11eb-852c-09c88f2f9afc.png)

## Environment
I leave my environment below. I tested it out on a single GPU.
<br><br>
### OS/GPU/CPU
* OS:
	* Linux(Ubuntu 18.04.5 LTS) 
* GPU:
	* NVIDIA® GeForce® RTX 2080 Ti VENTUS 11GB OC
* CPU:
	* Intel® Xeon® CPU E5640 @ 2.67GHz

### Dependencies

* Python = 3.6
* PyTorch = 1.6
* scipy
* numpy
* plotly (only for plotting)
* matplotlib (only for plotting in GA)
* pandas (only for mean of test score)

### Docker(option)
1. build or pull docker image
<br><br>
build image
```bash
./docker.sh build
```
pull image
```bash
docker pull docker4rintarooo/mdvrp:latest
```
<br><br>
2. run container using docker image(-v option is mount directory)
```bash
./docker.sh run
```

## Usage
* train
* inference(with 10 data)
* inference(with 1 data)

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
 
### inference phase(with 10 test data using shell script)

run `test.sh`.

`g` option generates 10 test data.
```
./test.sh g
```
If you create data by mistake, you can remove them with `rm` option
```
./test.sh rm
```


Now you can test with `or`, `to` and `ga` option.  
You can see the result score in `Csv` dir.  


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