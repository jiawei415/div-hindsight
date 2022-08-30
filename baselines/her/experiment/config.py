from copy import deepcopy
import numpy as np
import json
import os
import gym
from baselines import logger
from baselines.her.ddpg import DDPG
from baselines.her.her import make_sample_her_transitions, \
                              make_sample_her_transitions_diversity, \
                              make_sample_her_transitions_diversity_with_kdpp
from bher.common.monitor import Monitor
from bher.envs.multi_world_wrapper import PointGoalWrapper, SawyerGoalWrapper, ReacherGoalWrapper
# some params
DEFAULT_ENV_PARAMS = {
    'Point2D':
        {'n_cycles': 1, 'batch_size': 64, 'n_batches': 5, 'subset_size':100},
    'SawyerReach':
        {'n_cycles': 5, 'batch_size': 64, 'n_batches': 5, 'subset_size':100},
    'FetchReach':
        {'n_cycles': 5, 'batch_size': 64, 'n_batches': 5, 'subset_size':100},
    'Reacher-v2':
        {'n_cycles': 15, 'batch_size': 64, 'n_batches': 5, 'subset_size':100},
    'SawyerDoorPos-v1':
        {'n_cycles': 10, 'batch_size': 64, 'n_batches': 5, 'subset_size':100},
    'SawyerDoorAngle-v1':
        {'n_cycles': 20, 'batch_size': 64, 'n_batches': 5, 'subset_size':100},
    'SawyerDoorFixEnv-v1':
        {'n_cycles': 50, 'batch_size': 256, 'n_batches': 40, 'subset_size':300},
    'PointMass':
        {'n_cycles': 50, 'batch_size': 256, 'n_batches': 40, 'subset_size':300},
    'Fetch':
        {'n_cycles': 50, 'batch_size': 256, 'n_batches': 40, 'subset_size':300},
    'Hand':
        {'n_cycles': 50, 'batch_size': 256, 'n_batches': 40, 'subset_size':300},
}


DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    'max_episode_steps': 50,
    # ddpg
    'k_heads': 1,
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # int(1E6) int(1E6) bug for experience replay 
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,
    # training
    'n_cycles': 50,  # per epoch
    'rollout_batch_size': 1,  # per mpi thread
    'n_batches': 40,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
    # random init episode
    'random_init': 20,

    # prioritized_replay (tderror) has been removed
    'alpha': 0.6, # 0.6
    'beta0': 0.4, # 0.4
    'beta_iters': None, # None
    'eps': 1e-6,
}

CACHED_ENVS = {}
def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]

def prepare_params(kwargs):
    # default max episode steps
    default_max_episode_steps = 50
    # DDPG params
    ddpg_params = dict()

    env_name = kwargs['env_name']
    def make_env(subrank=None):
        try:
            env = gym.make(env_name, rewrad_type='sparse')
        except:
            logger.log('Can not make sparse reward environment')
            env = gym.make(env_name)
        # add wrapper for multiworld environment
        if env_name.startswith('Point'):
            env = PointGoalWrapper(env)
        elif env_name.startswith('Sawyer'):
            env = SawyerGoalWrapper(env)
        elif env_name.startswith('Reacher'):
            env = ReacherGoalWrapper(env)
        max_episode_steps = default_max_episode_steps
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        if (subrank is not None and logger.get_dir() is not None):
            try:
                from mpi4py import MPI
                mpi_rank = MPI.COMM_WORLD.Get_rank()
            except ImportError:
                MPI = None
                mpi_rank = 0
                logger.warn('Running with a single MPI process. This should work, but the results may differ from the ones publshed in Plappert et al.')
            env =  Monitor(env, os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)), allow_early_resets=True)
        return env

    kwargs['make_env'] = make_env
    kwargs['T'] = kwargs['max_episode_steps']
    kwargs['max_u'] = np.array(kwargs['max_u']) if type(kwargs['max_u']) == list else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak', 
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals',
                 'alpha', 'beta0', 'beta_iters', 'eps', 'k_heads']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['ddpg_params'] = ddpg_params

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    def reward_fun(ag_2, g, info={}):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
    }
    for name in ['replay_strategy', 'replay_k']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]
        del params[name]
    if params['prioritization'] == 'diversity':
        """
        decide if use the kdpp to sample
        """
        if params['use_kdpp']:
            # use subset size
            her_params['subset_size'] = params['subset_size']
            params['_' + 'subset_size'] = her_params['subset_size']
            del params['subset_size']
            her_params['goal_type'] = params['goal_type']
            her_params['sigma'] = params['sigma']
            sample_her_transitions = make_sample_her_transitions_diversity_with_kdpp(**her_params)
        else:
            sample_her_transitions = make_sample_her_transitions_diversity(**her_params)
    else:
        sample_her_transitions = make_sample_her_transitions(**her_params)
    return sample_her_transitions

def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b

def configure_ddpg(dims, params, reuse=False, use_mpi=True, clip_return=True):
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    gamma = params['gamma']
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params['ddpg_params']
    prioritization = params['prioritization']
    env_name = params['env_name']
    max_timesteps = params['max_timesteps']
    input_dims = dims.copy()
    # DDPG agent
    env = cached_make_env(params['make_env'])
    env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'gamma': gamma,
                        'prioritization': prioritization,
                        'env_name': env_name,
                        'max_timesteps': max_timesteps,
                        })
    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }
    # if use diversity
    if params['prioritization'] == 'diversity':
        ddpg_params['goal_type'] = params['goal_type']
    policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=use_mpi)
    return policy

def configure_dims(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())
    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
    }
    # for key, value in info.items():
    #     value = np.array(value)
    #     if value.ndim == 0:
    #         value = value.reshape(1)
    #     dims['info_{}'.format(key)] = value.shape[0]
    return dims