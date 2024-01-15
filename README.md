# Provably Efficient Exploration in Constrained Reinforcement Learning: Posterior Sampling Is All You Need

```
rm -r venv
python3 -m venv venv
source venv/bin/activate 
pip3 install -r requirements.txt 
pip3 install git+https://github.com/timvieira/arsenal.git 
```


## Commands to reproduce results in our paper:

```$ cd src```

### marsrover 4x4
```$ python -u run.py --alg cucrl_transitions posterior_transitions cucrl_pessimistic cucrl_optimistic --env gridworld --rounds 9000 --num_runs 100```


### marsrover 8x8
```$ python -u run.py --alg posterior_transitions cucrl_pessimistic cucrl_optimistic --env marsrover_gridworld --rounds 20000 --num_runs 30```

### box 4x4
```$ python -u run.py --alg posterior_transitions cucrl_pessimistic cucrl_optimistic --env box_gridworld --rounds 500000 --num_runs 30```
