# Sample efficient exploration via model-based reinforcement learning

This repository holds the source code to our paper [An Empirical Evaluation of Posterior Sampling for Constrained Reinforcement Learning](https://arxiv.org/abs/2209.03596).

# Reproducibility

Commands to reproduce results in our paper:

### marsrover 4x4
- `python -u run.py --alg lagr_posterior posterior_transitions cucrl_pessimistic cucrl_optimistic --env gridworld --rounds 9000 --num_runs 100 --bonus_coef 0.01 --horizon 20`
- `python -u run.py --alg cucrl_transitions --env gridworld --rounds 9000 --num_runs 30 --bonus_coef 0.01 --horizon 20`

### marsrover 8x8
- `python -u run.py --alg lagr_posterior posterior_transitions cucrl_pessimistic cucrl_optimistic --env marsrover_gridworld --rounds 200000 --num_runs 30 --bonus_coef 0.01 --horizon 1000`

### box 4x4
- `python -u run.py --alg lagr_posterior posterior_transitions cucrl_pessimistic cucrl_optimistic --env box_gridworld --rounds 2000000 --num_runs 30 --bonus_coef 0.5 --horizon 1000`


# Paper

If you use our code in your research, please remember to cite our paper:

```
@misc{Provodin_PS_CMDP_2022,
  url = {https://arxiv.org/abs/2209.03596},
  author = {Provodin, Danil and Gajane, Pratik and Pechenizkiy, Mykola and Kaptein, Maurits},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {An Empirical Evaluation of Posterior Sampling for Constrained Reinforcement Learning},
  publisher = {arXiv},
  year = {2022}
}
```
