# MDVRP solver with Multi-Head Attention

## Dependencies

* Python = 3.6
* PyTorch = 1.6
* scipy
* numpy
* plotly (only for plotting)
* matplotlib (only for plotting)


## Usage

### train

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

### inference
Generate test data
  
(GA -> txt file, Torch and Ortools -> json file).

```
python dataclass.py
```

Plot prediction of the pretrained model
```
cd Torch && python plot.py -p Weights/***.pt -t data/***.json -b 128
```
```
cd GA && python main.py data/***.txt
```
```
cd Ortools && python main.py -p data/***.json
```

## Reference
### DRL_MHA
* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/d-eremeev/ADM-VRP
* https://qiita.com/ohtaman/items/0c383da89516d03c3ac0

### Google Or-Tools
* https://github.com/skatsuta/vrp-solver

### GA(Python)
* https://github.com/Lagostra/MDVRP

### GA(C++)
* https://github.com/mathiasaap/Multi-Depot-Routing---Genetic-algorithm
