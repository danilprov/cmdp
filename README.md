# Sample efficient exploration via model-based reinforcement learning

This repository holds the source code to our paper [An Empirical Evaluation of Posterior Sampling for Constrained Reinforcement Learning](https://arxiv.org/abs/2209.03596).

# Reproducibility

1. create venv

```$ virtualenv myproject source myproject/venv/bin/activate```
2. clone repository

```$ git clone https://github.com/danilprov/cmdp.git```
3. install requirements

```$ pip -r requirements.txt```

4. install `arsenal` manually

```
$ pip install -r https://raw.githubusercontent.com/timvieira/arsenal/master/requirements.txt
$ pip install git+https://github.com/timvieira/arsenal.git 
```

## Commands to reproduce results in our paper:

```$ cd src```

### marsrover 4x4
```$ python -u run.py --alg cucrl_transitions posterior_transitions cucrl_pessimistic cucrl_optimistic --env gridworld --rounds 9000 --num_runs 100 --bonus_coef 0.1```


### marsrover 8x8
```$ python -u run.py --alg posterior_transitions cucrl_pessimistic cucrl_optimistic --env marsrover_gridworld --rounds 50000 --num_runs 30 --bonus_coef 0.1```

### box 4x4
```$ python -u run.py --alg posterior_transitions cucrl_pessimistic cucrl_optimistic --env box_gridworld --rounds 500000 --num_runs 30 --bonus_coef 0.1```


# Paper

If you use our code in your research, please remember to cite our paper:

```LaTeX
@misc{Provodin_PS_CMDP_2022,
  url = {https://arxiv.org/abs/2209.03596},
  author = {Provodin, Danil and Gajane, Pratik and Pechenizkiy, Mykola and Kaptein, Maurits},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {An Empirical Evaluation of Posterior Sampling for Constrained Reinforcement Learning},
  publisher = {arXiv},
  year = {2022}
}
```
