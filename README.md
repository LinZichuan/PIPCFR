## Requirements and setup

You will need to install:
- geomloss==0.2.4
- numpy
- pandas
- scikit_learn==1.0.2
- scikit_uplift==0.4.0
- torch==1.10.1
- tensorboard
- protobuf==3.20
- scikit-optimize
- ot==0.9.5
- tensorflow-gpu==1.15.0
- causalml


The dependencies for Pairnet and ESCFR from github:

```bash
git clone https://github.com/nlokeshiisc/pairnet_release.git
```

## Generate Data

**IHDP**
```
cd data/IHDP
python generate_data.py
cd ..
```

**NEWS**
```
cd data/NEWS
python generate_data.py
cd ..
```

**Synthetic Data**
```
cd data/Synthetic
bash generate.sh
cd ..
```

## Run the Experiments

Run experiments for NEWS dataset
```
MODELS_JSON='./examples/model_news.json' python run_experiment.py
```

Run experiments for IHDP dataset
```
MODELS_JSON='./examples/model_ihdp.json' python run_experiment.py
```

Run experiments for Synthetic dataset
```
MODELS_JSON='./examples/model_synthetic.json' python run_experiment.py
```

`MODELS_JSON` is the configure information of models. You can add model in `./examples/`. 
