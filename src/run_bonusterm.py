import argparse
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np

from codebase.algs.psrl2 import PSRLOptimistic, PSRLTransitions, PSRLLagrangian, CUCRLOptimistic
from codebase.algs.c_ucrl2 import CUCRLConservative, CUCRLOptimistic2, CUCRLTransitions
from codebase.environments.gridworld import GridWorld
from codebase.environments.box_gridworld import BoxGridWorld
from codebase.mdp import FiniteHorizonCMDP
from codebase.rl_solver.lin_prog import LinProgSolver
from codebase.rl_solver.nonlin_prog import NonlinProgSolver
from codebase.rl_solver.planner import ValueIteration, RelativeValueIteration
from codebase.args import get_args

parser = argparse.ArgumentParser()
get_args(parser)
args = parser.parse_args()

default_maps = {'box_gridworld': '6x6_box',
                'marsrover_gridworld': '8x8_marsrover',
                'gridworld': '4x4_gwsc'}

if args.map == 'default':
    args.map = default_maps[args.env]


def train(alg, args):
    results = np.zeros((args.num_runs, args.rounds, args.d + 1))

    for run in tqdm(range(1, args.num_runs + 1)):
        alg.reset()
        run_results = np.zeros((1, args.d + 1))
        while True:
            rewards, term = alg()
            run_results = np.vstack((run_results, rewards))
            if term:
                break
        results[run - 1, :, :] = run_results[1:, :]

    return results


def main():
    if args.env == 'gridworld':
        args.horizon = 20
        budget = [0.2]
        G = GridWorld(args=args)
    elif args.env == 'marsrover_gridworld':
        budget = [0.1]
        args.horizon = 20
        G = GridWorld(args=args)
    elif args.env == 'box_gridworld':
        budget = [0.6]
        args.horizon = 1000
        G = BoxGridWorld(args=args)

    now = datetime.now()
    date = now.strftime("%Y%m%d%H%M%S")
    model_dir = f'{args.output_dir}/{date}_{args.env}_{args.map}_{args.rounds}'
    # model_dir = f'{args.output_dir}/{date}_{args.env}_{args.map}_{args.rounds}'
    if not os.path.exists(model_dir):
        print(f'Creating a new model directory: {model_dir}')
        os.makedirs(model_dir)

    bonuses = [0.01, 0.05, 0.1, 0.2, 0.5]
    for bonus in bonuses:
        args.bonus_coef = bonus

        [mdp_values, Si, Ai] = G.encode()  # [MDP, State-lookups, Action-lookups]
        args.num_states = G.num_states
        args.rows = G.rows
        args.cols = G.cols
        args.initial_states = G.initial_states
        d = len(budget)
        args.d = d
        env = FiniteHorizonCMDP(*mdp_values, d, budget, G.H, Si, Ai, G.terminals)

        for alg_name in args.algs:
            if alg_name == 'posterior_transitions':
                planner = LinProgSolver(M=env, args=args)
                alg = PSRLTransitions(G=G, M=env, args=args, planner=planner)
            elif alg_name == 'posterior_rewards':
                planner = LinProgSolver(M=env, args=args)
                alg = PSRLOptimistic(G=G, M=env, args=args, planner=planner)
            elif alg_name == 'cucrl_conservative':
                planner = LinProgSolver(M=env, args=args)
                alg = CUCRLConservative(G=G, M=env, args=args, planner=planner)
            elif alg_name == 'cucrl_optimistic':
                planner = LinProgSolver(M=env, args=args)
                alg = CUCRLOptimistic(G=G, M=env, args=args, planner=planner)
            elif alg_name == 'cucrl_optimistic2':
                planner = LinProgSolver(M=env, args=args)
                alg = CUCRLOptimistic2(G=G, M=env, args=args, planner=planner)
            elif alg_name == 'cucrl_transitions':
                planner = NonlinProgSolver(M=env, args=args)
                alg = CUCRLTransitions(G=G, M=env, args=args, planner=planner)
            elif alg_name == 'lagr_posterior':
                planner = RelativeValueIteration(M=env, args=args)
                alg = PSRLLagrangian(G=G, M=env, args=args, planner=planner)

            print(f'Running an experiment for algorithm: {alg_name} witn bonus term {args.bonus_coef}')
            results = train(alg, args)

            alg_name = alg.__class__.__name__
            output_filename = f'{alg_name}_{args.bonus_coef}'
            with open(model_dir + '/' + output_filename + '.pickle', 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

