def get_args(parser):
    # Generic hyperparamters
    parser.add_argument("--map", type=str, default='default')
    parser.add_argument('-alg', '--algs', nargs='+', default=['lagr_posterior'])
    parser.add_argument("--bonus_coef", type=float, default=0.01)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--seed", type=str, default='random')
    parser.add_argument("--solver", choices=['value_iteration', 'linear_prog'], default="value_iteration")
    parser.add_argument("--output_dir", type=str, default="./log")
    parser.add_argument("--env", choices=['box_gridworld', 'gridworld', 'marsrover_gridworld'], default='box_gridworld')
    parser.add_argument("--budget", nargs="+", type=float, default=[0.2])
    parser.add_argument("--randomness", type=float, default=0.1)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--infinite", type=bool, default=True)
    parser.add_argument("--num_runs", type=int, default=30)

    args = parser.parse_args()
