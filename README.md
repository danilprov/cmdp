# Efficient Exploration in Constrained RL: Achieving Near-Optimal Regret With Posterior Sampling
Official code for the paper ["Efficient Exploration in Average-Reward Constrained Reinforcement Learning: Achieving Near-Optimal Regret With Posterior Sampling"](https://arxiv.org/abs/2405.19017)

Danil Provodin, Maurits Kaptein, Mykola Pechenizkiy.

## Commands to reproduce results in our paper:
### Setup
```
rm -r venv
python3 -m venv venv
source venv/bin/activate  (venv\Scripts\activate "for windows")
pip3 install -r requirements.txt 
pip3 install git+https://github.com/timvieira/arsenal.git 
```

For training run, `cd` to the `src` folder. Then run

### marsrover 4x4
```$ python -u run.py --alg posterior_transitions cucrl_conservative --env gridworld --rounds 11000 --num_runs 50```

```$ python -u run.py --alg cucrl_transitions fha_cmdp --env gridworld --rounds 11000 --num_runs 10```


### marsrover 8x8
```$ python -u run.py --alg posterior_transitions cucrl_conservative --env marsrover_gridworld --rounds 20000 --num_runs 50```

### box 4x4
```$ python -u run.py --alg posterior_transitions cucrl_conservative cucrl_optimistic --env box_gridworld --rounds 500000 --num_runs 30```

## Earlier versions
This repo also contains experiments of an earlier paper ["An Empirical Evaluation of Posterior Sampling for Constrained Reinforcement Learning"](https://arxiv.org/abs/2209.03596) presented at NeurIPS'22 RL4RealLife workshop. To access experiments from this paper please check out a git tag `git checkout tags/v1.2 -b <branch_name>`.

