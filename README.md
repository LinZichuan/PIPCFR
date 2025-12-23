# PIPCFR: Pseudo-outcome Imputation with Post-treatment Variables for Individual Treatment Effect Estimation

<!-- <a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> -->

[![Paper](https://img.shields.io/badge/Paper-Arxiv%20Link-light)](https://arxiv.org/abs/2512.18737)

## Installation

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

## Generate Training Data

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

## Run Experiments

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


## Citation

If you find this project useful in your research, please consider citing:

<!-- ```bibtex
@article{lin2025adapt,
  title={AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition},
  author={Zichuan Lin and Yicheng Liu and Yang Yang and Lvfang Tao and Deheng Ye},
  journal={arXiv preprint arXiv:2512.03794},
  year={2025}
}
``` -->

## License
- PIPCFR is licensed under the Apache License 2.0. 
