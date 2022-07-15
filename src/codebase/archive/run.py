import argparse
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

from codebase.algs.psrl import PosteriorSampling
from codebase.algs.c_ucrl import CUCRLAlgorithm
from codebase.algs.lagrangian_psrl import LagrangianPosteriorSampling
from codebase.environments.gridworld import GridWorld
from codebase.environments.box_gridworld import BoxGridWorld
from codebase.mdp import FiniteHorizonCMDP
from codebase.rl_solver.planner import ValueIteration
from codebase.rl_solver.lin_prog import LinProgSolver
from codebase.rl_solver.nonlin_prog import NonlinProgSolver
from codebase.args import get_args, set_gridworld_defaults

parser = argparse.ArgumentParser()
get_args(parser)
args = parser.parse_args()
set_gridworld_defaults(args)

default_maps = {'box_gridworld': '6x6_box',
                'marsrover_gridworld': '8x8_marsrover',
                'gridworld': '4x4_gwsc'}

if args.map == 'default':
    args.map = default_maps[args.env]

def train(alg, args):
    results = {'round': [], 'mixture_reward': [], 'mixture_constraint': [],
               'current_reward': [], 'current_constraint': [], 'policy': []}
    # for round in range(args.rounds):
    for round in tqdm(range(1, args.rounds + 1)):
        [metrics, status_str] = alg()
        metrics['round'] = round
        # results.append(metrics)
        for key in results.keys():
            results[key].append(metrics[key])
        # print(f'Round: {round}/{args.rounds}')
        # print(f'{status_str}')
        # print(f'----------------------------')

    return results


def main():
    if args.env == 'gridworld':
        G = GridWorld(args=args)
        budget = [0.2]
    elif args.env == 'marsrover_gridworld':
        G = GridWorld(args=args)
        budget = [0.1]
    elif args.env == 'box_gridworld':
        G = BoxGridWorld(args=args)
        budget = [0.2]

    [mdp_values, Si, Ai] = G.encode()  # [MDP, State-lookups, Action-lookups]
    args.num_states = G.num_states
    args.rows = G.rows
    args.cols = G.cols
    args.initial_states = G.initial_states
    d = len(budget)
    env = FiniteHorizonCMDP(*mdp_values, d, budget, G.H, Si, G.terminals)

    now = datetime.now()
    date = now.strftime("%Y%m%d%H%M%S")
    model_dir = f'{args.output_dir}/{date}_{args.env}_{args.map}_{args.horizon}'
    if not os.path.exists(model_dir):
        print(f'Creating a new model directory: {model_dir}')
        os.makedirs(model_dir)

    for alg_name in args.algs:
        print(f'Running an experiment for algorithm: {alg_name}')
        results = {}
        for run in range(args.num_runs):

            if alg_name == 'posterior_transitions':
                args.posterior_type = 'transitions'
                planner = LinProgSolver(M=env, args=args)
                alg = PosteriorSampling(G=G, M=env, args=args, planner=planner)
            elif alg_name == 'posterior_rewards':
                args.posterior_type = 'rewards'
                planner = LinProgSolver(M=env, args=args)
                alg = PosteriorSampling(G=G, M=env, args=args, planner=planner)
            elif alg_name == 'posterior_full':
                args.posterior_type = 'full'
                planner = LinProgSolver(M=env, args=args)
                alg = PosteriorSampling(G=G, M=env, args=args, planner=planner)
            elif alg_name == 'cucrl_pessimistic':
                args.ucb_type = 'cucrl_pessimistic'
                planner = LinProgSolver(M=env, args=args)
                alg = CUCRLAlgorithm(G=G, M=env, args=args, planner=planner)
            elif alg_name == 'cucrl_optimistic':
                args.ucb_type = 'cucrl_optimistic'
                planner = LinProgSolver(M=env, args=args)
                alg = CUCRLAlgorithm(G=G, M=env, args=args, planner=planner)
            elif alg_name == 'cucrl_transitions':
                args.ucb_type = 'cucrl_transitions'
                planner = NonlinProgSolver(M=env, args=args)
                alg = CUCRLAlgorithm(G=G, M=env, args=args, planner=planner)
            elif alg_name == 'lagr_posterior':
                planner = ValueIteration(M=env, args=args)
                alg = LagrangianPosteriorSampling(G=G, M=env, args=args, planner=planner)

            run_results = train(alg, args)
            results['run' + str(run)] = run_results

            output_filename = f'{alg_name}'
            with open(model_dir + '/' + output_filename + '.pickle', 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()


# python -u temp.py --alg posterior_rewards posterior_transitions cucrl_pessimistic cucrl_optimistic --env marsrover_gridworld --rounds 200
