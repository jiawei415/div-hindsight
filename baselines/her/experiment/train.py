import os
import sys
import click
import numpy as np
import json
from tqdm import tqdm
from mpi4py import MPI
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork
from baselines.common.env_util import build_env, get_game_envs

import os.path as osp
import tempfile
import datetime
import pickle

_game_envs = get_game_envs(print_out=False)
def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]

def train(policy, rollout_worker, evaluator, n_epochs, n_test_rollouts, n_cycles, n_batches,
          policy_save_interval, save_policies, num_cpu, dump_buffer, clip_div, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()
    # the path of the saved models
    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    rollout_num = kwargs["rollout_num"]
    # random_init for o/g/rnd stat and model training
    random_init = kwargs["random_init"]
    if random_init:
        logger.info('Random initializing ...')
        rollout_worker.clear_history()
        # rollout_worker.render = True
        random_num = int(random_init) // rollout_num * rollout_num
        for epi in range(random_num):
            episode = rollout_worker.generate_rollouts(random_ac=True)
            policy.store_episode(episode, dump_buffer, clip_div)

    # for the training
    logger.info("Training...")
    best_success_rate = -1
    t = 1
    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        for cycle in tqdm(range(n_cycles)):
            for _ in range(rollout_num):
                episode = rollout_worker.generate_rollouts()
                policy.store_episode(episode, dump_buffer, clip_div)
            for batch in tqdm(range(n_batches)):
            # for batch in range(n_batches):
                #print('[{}] Epoch: {}, Cycle: {}, Batch: {}'.format(datetime.datetime.now(), epoch, cycle, batch))
                t = ((epoch*n_cycles*n_batches)+(cycle*n_batches)+batch)*num_cpu
                policy.train(t, dump_buffer)
            policy.update_target_net()
        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts * rollout_num):
            evaluator.generate_rollouts()
        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))
        if rank == 0:
            print('[{}]'.format(datetime.datetime.now()))
            logger.dump_tabular()
            if dump_buffer:
                policy.dump_buffer(epoch)
        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)
        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

def launch(env, num_env,
    env_name, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return,
    prioritization, binding, logging, version, dump_buffer, n_cycles,
    clip_div, logdir, goal_type, use_kdpp, subset_size, sigma, override_params={}, save_policies=False):
    # Fork for multi-CPU MPI implementation.
    # if num_cpu > 1:
    #     whoami = mpi_fork(num_cpu, binding)
    #     if whoami == 'parent':
    #         sys.exit(0)
    #     import baselines.common.tf_util as U
    #     U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()
    # Configure logging
    if logging: 
        logdir = os.path.join(logdir, f"dher_{env_name}_{seed}")
        logdir = os.path.expanduser(logdir)
    else:
        logdir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    # if rank == 0:
    #     if logdir or logger.get_dir() is None:
    #         logger.configure(dir=logdir)
    # else:
    #     logger.configure()
    logger.configure(dir=logdir)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)
    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)
    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    params['prioritization'] = prioritization
    params['binding'] = binding
    params['max_timesteps'] = n_epochs * params['n_cycles'] *  params['n_batches'] * num_cpu
    params['version'] = version
    params['dump_buffer'] = dump_buffer
    params['n_cycles'] = n_cycles
    params['clip_div'] = clip_div
    params['n_epochs'] = n_epochs
    params['num_cpu'] = num_cpu
    params['goal_type'] = goal_type
    params['use_kdpp'] = use_kdpp
    params['subset_size'] = subset_size
    params['sigma'] = sigma
    # if dump the buffer
    if params['dump_buffer']:
        params['alpha'] = 0
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    if env_name.startswith('Point2D'):
        params.update(config.DEFAULT_ENV_PARAMS['Point2D'])
    if env_name.startswith('PointMass'):
        params.update(config.DEFAULT_ENV_PARAMS['PointMass'])
    elif env_name.startswith('FetchReach'):
        params.update(config.DEFAULT_ENV_PARAMS['FetchReach'])
    elif env_name.startswith('Fetch'):
        params.update(config.DEFAULT_ENV_PARAMS['Fetch'])
    elif env_name.startswith('SawyerReach'):
        params.update(config.DEFAULT_ENV_PARAMS['SawyerReach'])
    elif env_name.startswith('Hand'):
        params.update(config.DEFAULT_ENV_PARAMS['Hand'])
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter\
    params['n_batches'] *= (params['batch_size'] // 64)
    params['batch_size'] = 64
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        # json.dump(params, f)
        f.write(json.dumps(params, indent=4) + '\n')
        f.flush()
        f.close()
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)
    # check the dim
    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)
    # some params
    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }
    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }
    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]
    # rollout worker for sampling the trajectory
    rollout_worker = RolloutWorker(env, policy, dims, logger, **rollout_params)
    # rollout_worker.seed(rank_seed)
    # eval
    eval_env = env
    evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)
    # evaluator.seed(rank_seed)
    # train the algorithm
    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies,
        num_cpu=num_cpu, dump_buffer=dump_buffer, clip_div=clip_div,
        random_init=params['random_init'], rollout_num=16)

# some parameters
@click.command()
@click.option('--env_name', default='FetchPickAndPlace-v1', help='the name of envs')
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--num_env', type=int, default=1, help='Number of environment copies being run')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=0, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'final', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--prioritization', type=click.Choice(['none', 'diversity']), default='diversity', help='the prioritization strategy to be used.')
@click.option('--binding', type=click.Choice(['none', 'core']), default='none', help='configure mpi using bind-to none or core.')
@click.option('--logging', type=bool, default=True, help='whether or not logging')
@click.option('--version', type=int, default=0, help='version')
@click.option('--dump_buffer', type=bool, default=False, help='dump buffer contains data for analysis')
@click.option('--n_cycles', type=int, default=50, help='n_cycles')
@click.option('--clip_div', type=float, default=999, help='clip_diversity')
@click.option('--logdir', type=str, default='~/results/her', help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--goal_type', type=str, default='full', help='decide which kinds of goals are used for the training')
@click.option('--use_kdpp', type=bool, default=True, help='whether or not use kdpp')
@click.option('--subset_size', type=int, default=100, help='the subset size for the k-dpp')
@click.option('--sigma', type=float, default=0.5, help='the sigma of the rbf kernel, fetch use 0.5, and hand use 0.1')

def main(**kwargs):
    env = build_env(kwargs, _game_envs)
    kwargs.update({"env": env})
    launch(**kwargs)

if __name__ == '__main__':
    main()