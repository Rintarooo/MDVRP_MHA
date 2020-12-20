# MDVRP solver with Multi Heads Attention

TensorFlow2 and PyTorch implementation of ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!(Kool et al. 2019)(https://arxiv.org/pdf/1803.08475.pdf)

## Dependencies

* Python >= 3.6
* TensorFlow >= 2.0
* PyTorch = 1.5
* tqdm
* scipy
* numpy
* plotly (only for plotting)
* matplotlib (only for plotting)


## Usage

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

Plot prediction of the pretrained model

```
python plot.py -p Weights/***.pt(or ***.h5)
```

If you want to verify your model, you can generate data(GA -> txt file, Torch, Ortools -> json file).

```
python plot.py -p Weights/***.pt -t data/.json -b 128
```

## Reference
* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/d-eremeev/ADM-VRP
* https://qiita.com/ohtaman/items/0c383da89516d03c3ac0