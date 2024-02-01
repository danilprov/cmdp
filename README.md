# Provably Efficient Exploration in Constrained Reinforcement Learning: Posterior Sampling Is All You Need

```
rm -r venv
python3 -m venv venv
source venv/bin/activate  (venv\Scripts\activate "for windows")
pip3 install -r requirements.txt 
pip3 install git+https://github.com/timvieira/arsenal.git 
```


## Commands to reproduce results in our paper:

```$ cd src```

### marsrover 4x4
```$ python -u run.py --alg posterior_transitions cucrl_conservative --env gridworld --rounds 11000 --num_runs 50```

```$ python -u run.py --alg cucrl_transitions fha_cmdp --env gridworld --rounds 11000 --num_runs 10```


### marsrover 8x8
```$ python -u run.py --alg posterior_transitions cucrl_conservative --env marsrover_gridworld --rounds 20000 --num_runs 50```

### box 4x4
```$ python -u run.py --alg posterior_transitions cucrl_conservative cucrl_optimistic --env box_gridworld --rounds 500000 --num_runs 30```
