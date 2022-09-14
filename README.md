# Sample efficient exploration via model-based reinforcement learning

Source code to our paper [An Empirical Evaluation of Posterior Sampling for Constrained Reinforcement Learning](https://arxiv.org/abs/2209.03596)

## running the code

Commands to reproduce results in our paper:

### marsrover 4x4
- `python -u run.py --alg lagr_posterior posterior_transitions cucrl_pessimistic cucrl_optimistic --env gridworld --rounds 9000 --num_runs 100 --bonus_coef 0.01 --horizon 20`
- `python -u run.py --alg cucrl_transitions --env gridworld --rounds 9000 --num_runs 30 --bonus_coef 0.01 --horizon 20`

### marsrover 8x8
- `python -u run.py --alg lagr_posterior posterior_transitions cucrl_pessimistic cucrl_optimistic --env marsrover_gridworld --rounds 200000 --num_runs 30 --bonus_coef 0.01 --horizon 1000`

### box 4x4
- `python -u run.py --alg lagr_posterior posterior_transitions cucrl_pessimistic cucrl_optimistic --env box_gridworld --rounds 2000000 --num_runs 30 --bonus_coef 0.5 --horizon 1000`
