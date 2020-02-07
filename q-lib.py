"""Q-Learning module.
    
    This module implements different Q-Learning algorithms and the
    approximators used in this algorithms, e.g. Q-tables and
    Q-function approximators. Additionally, some schedules, an
    indexing class, and a replay memory class are implemented.

    Standard usage of this module only loads the specific Q-Learning
    algorithm you want to use, e.g. QL or DQL and use the public
    interface:
        __init__: Initialize the learning algorithm.
        info: Print the set parameters and all information of the
            learning algorithm to the logging stream.
        learn: Train the learning agent, with the passed arguments.
            This method also takes safety algorithms as an argument.
        save: Save the trained agent to the disk, including all
            learning parameters.
        load: Load a trained agent to an empty instance of an agent.

    For advanced usage, load the objects you want to manipulate from
    the model and have a look at the object's documentation .
"""
import numpy as np
from numpy import newaxis
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
import dill as pickle
from collections import namedtuple
import random
from itertools import count
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.utils import seeding

""" Logger Setup """
# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create STDERR handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
ch.setFormatter(formatter)

# Set STDERR handler as the only handler
logger.handlers = [ch]

# prevent propagation
logger.propagate = False


class BaseQ(object):
    """Base class for Q-Function approximators.

        This class acts as a parent class to the two specific Q-function
        approximator classes implemented below. It implements the methods
        shared by both classes and defines the basic interface.
    """
    def __init__(self, np_random):
        """Initialize base class.
        
            This method initialzes the base class. The only attribute which is
            set, is the random number generator. The method needs to be
            implemented for any specific approximator separately.

            Args:
                np_random (RandomState): State of a numpy random generator.
                    This attribute is used for any random action, within this
                    calls to enable reproducability.
        """
        self.np_random = np_random
    
    def __call__(self, state):
        """Call interface.

            This method allows to evaluate the Q-function approximator by
            directly calling the class instance. For this, the _eval method
            needs to be implemented.
        """
        return self._eval(state)

    def _eval(self, state):
        """Evaluation method.
        
            This method implements the evaluation of the Q-function
            approximator. It is private, since it should be called only
            by the internal __call__ method. The method needs to be
            implemented for any specific approximator separately.

            Args:
                state (np.array): Numpy array of the current state.
                    This can contain multiple states or just a single one.

            Returns:
                Q (np.array): Q-Values for the given state(s). This is always
                    an array, since for a single state, multiple Q-values
                    (corresponding to the different actions) are returned.

            Raises:
                NotImplementedError: If the generic base class is called.
        """
        raise NotImplementedError

    def greedy_action(self, state):
        """Greedy action selection.
        
            This method implements the greedy algorithm. Given the state
            the Q-function is evaluated and subsequently the action
            with the largest Q-value is returned.

            Args:
                state (np.array): Numpy array of the current state.
                    This can contain multiple states or just a single one.

            Returns:
                action (np.array): Action with the largest Q-value given
                    the current state. If multiple states are passed to
                    the method, multiple actions are returned.

            Note:
                If the largest Q-value is achieved by multiple actions, one
                of them is chosen randomly.
        """
        Q = self._eval(state)
        action_idx = self.np_random.choice(np.argwhere(np.max(Q) == Q)[:,0])
        return self._action_idx.reverse(action_idx)

    def update(self, batch):
        """Update method.
        
            This method updates the learning Q-function given the samples
            in the batch argument. The update is specific to the approximator
            used.

            Args:
                batch (np.array): Numpy array of the current state.
                    This can contain multiple states or just a single one.

            Raises:
                NotImplementedError: If the generic base class is called.
        """
        raise NotImplementedError

    def set_learning_parameters(self):
        """Set learning parameters.
        
            This method sets all specific parameters of the learning
            algorithm. Since the parameters are specific, this method
            needs to be implemented for each approximator separately.

            Args:
                None.

            Raises:
                NotImplementedError: If the generic base class is called.
        """
        raise NotImplementedError
        
    def copy(self):
        """Copy method.
        
            This method copies the learning Q-function to the target Q-function.
            This is done, in order to stabilize the learning process. See
            for more information.
            The method needs to be implemented for any specific approximator
            separately.

            Args:
                None.

            Returns:
                Nothing.

            Raises:
                NotImplementedError: If the generic base class is called.
        """
        raise NotImplementedError

    def export(self):
        """Export method.
        
            This method exports the learning Q-function and is used to save
            a trained agent, i.e., the learned Q-function.

            Args:
                None.

            Returns:
                Q: Instance of the Q approximator with learned parameters.

            Raises:
                NotImplementedError: If the generic base class is called.
        """
        raise NotImplementedError

    def import_(self, agent):
        """Import method.
        
            This method imports a previously exported and/or trained Q
            approximator instance. The passed instance is stored as the
            appropriate class attribute. Principally, this method can be
            used to warm start the learning process.

            Args:
                agent: Instance of the Q approximator, which should be
                    imported.

            Raises:
                NotImplementedError: If the generic base class is called.

            Note:
                The trailing underscore is needed since import is a generic
                Python call.
        """
        raise NotImplementedError


class QTable(BaseQ):
    """Q-Table approximator.

        This class implements the stanard Q-table approximator for Q-Learning.
        It uses an additional Q-table as a target function and utilizes
        a discretization object to discretize any continous system.
    """
    def __init__(self, env, discretization=[100, 3], custom_bounds={}, *args):
        """Initialize Q-table.
        
            This method initialzes the Q-table class.

            Args:
                env (gym.Env): Instance of an OpenAI gym environment. Possibly,
                    one of the models in models.py.
                discretization (list): List of the number of bins, to discretize
                    the state space and action space into.
                custom_bounds (dict): Dictionary containing OpenAI gym box objects.
                    Those box objects are either observation spaces or action spaces,
                    defining custom bounds for infinitely bounded states or actions.
                args: Generic arguments.
        """
        super(QTable, self).__init__(*args)
        # observation space
        self._d = [[], []]
        if type(env.observation_space) is Discrete:
            self._d[0] = [env.observation_space.n]
        else:
            if type(discretization[0]) is int:
                self._d[0] = [discretization[0]]*env.observation_space.shape[0]
            else:
                self._d[0] = discretization[0]
            assert len(self._d[0]) == env.observation_space.shape[0]

        # action space
        if type(env.action_space) is Discrete:
            self._d[1] = [env.action_space.n]
        else:
            if type(discretization[1]) is int:
                self._d[1] = [discretization[1]]*env.action_space.shape[0]
            else:
                self._d[1] = discretization[1]
            assert len(self._d[1]) == env.action_space.shape[0]

        if 'observation_space' in custom_bounds:
            assert type(custom_bounds['observation_space']) == type(env.observation_space)
            self._idx = Idx(custom_bounds['observation_space'], self._d[0])
        else:
            self._idx = Idx(env.observation_space, self._d[0])
        if 'action_space' in custom_bounds:
            assert type(custom_bounds['action_space']) == type(env.action_space)
            self._action_idx = Idx(custom_bounds['action_space'], self._d[1])
        else:
            self._action_idx = Idx(env.action_space, self._d[1])

        self.shape = tuple(self._d[0] + self._d[1])

        self._Q = np.zeros(self.shape)
        self._Q_target = np.zeros(self.shape)
        self._is_state = env.observation_space.contains

    def _eval(self, state):
        """Evaluation method.
        
            This method implements the evaluation of the Q-table.
            For more information see the generic method.

            Args:
                state (np.array): Numpy array of the current state.
                    This can contain multiple states or just a single one.

            Returns:
                Q (np.array): Q-table entries for the given state(s). This is
                    always an array, since for a single state, multiple Q-values
                    (corresponding to the different actions) are returned.
        """
        idx = self._idx(state)
        return self._Q[tuple(idx)]

    def update(self, batch):
        """Update method.
        
            This method updates the learning Q-function given the samples
            in the batch argument. The update is specific to the approximator
            used.

            For the learning algorithm using Q-tables, the update method
            computes the indexes of the states and actions and updates the
            corresponding entries. Making use of the current learning rate
            value and the target Q-table for the Q-values of the next state.

            Args:
                batch (np.array): Numpy array of the current state.
                    This can contain multiple states or just a single one.
        """
        batch = Transition(*zip(*batch))
        
        valid_states = np.empty(len(batch.done), dtype=np.bool)
        for i,d in zip(range(len(batch.done)), batch.done):
            if d:
                valid_states[i] = self._is_state(batch.next_state[i])
            else:
                valid_states[i] = not d
        
        state_idx = self._idx(np.array(batch.state))
        action_idx = self._action_idx(np.array(batch.action))
        next_state_idx = self._idx(np.array(batch.next_state)[valid_states])
        rewards = np.array(batch.reward)

        Q = self._Q[tuple(state_idx+action_idx)]
        
        if Q.shape:
            next_Q = np.zeros(Q.shape)
        else:
            next_Q = np.zeros(1)
        next_Q[valid_states] = np.max(self._Q_target[tuple(next_state_idx)], -1)

        Q_target = rewards + self._gamma * next_Q
        
        self._Q[tuple(state_idx+action_idx)] += self._alpha() * (Q_target - Q)

    def copy(self):
        """Copy method.
        
            This method copies the learning Q-function to the target Q-function.
            In the case of the Q-table approximator, this is a simple allocation.
            See the generic method for further information.

            Args:
                None.

            Returns:
                Nothing.
        """
        self._Q_target = self._Q

    def set_learning_parameters(self, gamma, num_episodes, alpha=None, alpha_0=0.5, decay_fraction=0.8, alpha_end=0.0,
        epsilon=None, epsilon_0=1.0, explore_fraction=0.1, final_epsilon=0.02):
         """Set learning parameters.
        
            This method sets the learning parameters of the Q-Learning
            algortihm with Q-tables as approximators.

            Args:
                gamma (float): Discout factor, between zero and one.
                num_episodes (int): Number of training episodes.
                alpha (Schedule): Learning rate schedule object. Defines
                    the schedule according to which the learning rate
                    is decayed. If None, a linear schedule is defined
                    using the next few arguments.
                alpha_0 (float): Starting value for learning rate
                    schedule.
                decay_fraction (float): Fraction of episodes over which
                    the learning rate is decayed.
                alpha_end (float): End value for the learning rate
                    schedule.
                epsilon (Schedule): Exploration schedule object. Defines
                    the schedule according to which the exploration fraction
                    is decayed. If None, a linear schedule is defined
                    using the next few arguments.
                epsilon_0 (float): Starting value for exploration
                    schedule.
                explore_fraction (float): Fraction of episodes over which
                    the exploration fraction is decayed.
                final_epsilon (float): End value for the exploration
                    schedule.
        """
        self._gamma = gamma
        if alpha is None:
            self._alpha = LinearSchedule(alpha_0, decay_fraction, alpha_end, num_episodes)
        else:
            assert Schedule in type(alpha).__bases__
            self._alpha = alpha
        if epsilon is None:
            self._epsilon = LinearSchedule(epsilon_0, explore_fraction, final_epsilon, num_episodes)
        else:
            assert Schedule in type(epsilon).__bases__
            self._epsilon = epsilon

    def export(self):
        """Export method.
        
            This method exports the learning Q-function and is used to save
            a trained agent, i.e., the learned Q-function. In the case of a
            Q-table, this method returns a numpy array.

            Args:
                None.

            Returns:
                Q (np.array): Numpy array containing the Q-table.
        """
        return self._Q

    def import_(self, agent):
        """Import method.
        
            This method imports a previously exported and/or trained Q
            approximator instance. For more information see the generic
            method.

            Args:
                agent (np.array): Numpy array containing a Q-table.
        """
        self._Q = agent


class QFunction(BaseQ):
    """PyTorch approximator.

        This class implements a Q-function approximator, given any functional
        representation as a PyTorch object. Most commonly this function
        approximator is a neural network, but could be something else as well.
    """
    def __init__(self, env, function_approximator, discretization=3, init_function=None, device=None, *args):
        """Initialize Q-function.
        
            This method initialzes the Q-function class.

            Args:
                env (gym.Env): Instance of an OpenAI gym environment. Possibly,
                    one of the models in models.py.
                function_approximator (PyTorch object): Function approximator defined
                    by a PyTorch object instance.
                discretization (int or list): (List of) number of bins, to discretize
                    the action space into. Due to the usage of a continuous function
                    approximator the state doesn't need to be discretized.
                init_function (PyTorch object): Optional initialize function applied to
                    layered networks. PyTorch enables specific NN initialization with
                    function, this is enabled with this argument.
                device (str): Wether to use the CPU or GPU to train and evaluate PyTorch
                    objects. If None, the GPU is chosen if one is available.
                args: Generic arguments.
        """
        super(QFunction, self).__init__(*args)
        if device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

        # action space
        if type(env.action_space) is Discrete:
            self._d = [env.action_space.n]
        else:
            if type(discretization) is int:
                self._d = [discretization]*env.action_space.shape[0]
            else:
                self._d = discretization
            assert len(self._d) == env.action_space.shape[0]

        self._action_idx = Idx(env.action_space, self._d)
        self._Q = function_approximator(env.observation_space.shape[0], np.sum(self._d)).to(device)
        self._Q_target = function_approximator(env.observation_space.shape[0], np.sum(self._d)).to(device)
        if init_function is not None:
            self._Q.apply(init_function)
            self._Q_target.apply(init_function)
        self._is_state = env.observation_space.contains

    def _eval(self, state):
        """Evaluation method.
        
            This method implements the evaluation of the Q-function.
            For more information see the generic method.

            Args:
                state (np.array): Numpy array of the current state.
                    This can contain multiple states or just a single one.

            Returns:
                Q (np.array): Q-function value for the given state(s). This is
                    always an array, since for a single state, multiple Q-values
                    (corresponding to the different actions) are returned.
        """
        return self._Q(torch.from_numpy(state).type(torch.FloatTensor).to(
            self._device)).cpu().detach().numpy()

    def update(self, batch):
        """Update method.
        
            This method updates the learning Q-function given the samples
            in the batch argument. The update is specific to the approximator
            used.

            For the learning algorithm using function approximators, this
            method makes heavy use of the PyTorch framework. All torch
            specific code is used in this method. For large batches, it makes
            sense to use a GPU for this operations, which can be specified in
            the __init__ method. For the action, the indexes are calculated
            using the index class defined at the end of this module.

            Args:
                batch (np.array): Numpy array of the current state.
                    This can contain multiple states or just a single one.
        """

        # transpose and preprocess
        batch = Transition(*zip(*batch))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is False, batch.done)), dtype=torch.uint8)
        non_final_next_states = torch.Tensor(batch.next_state).type(torch.FloatTensor)[non_final_mask]
        state_batch = torch.Tensor(batch.state).type(torch.FloatTensor)
        action_batch = torch.tensor(self._action_idx(np.array(batch.action))).view(-1, 1)
        reward_batch = torch.tensor(batch.reward).view(-1, 1)

        Q = self._Q(state_batch).gather(1, action_batch)

        next_Q = torch.zeros(Q.size(0))
        next_Q[non_final_mask] = self._Q_target(non_final_next_states).max(1)[0].detach()
        next_Q = next_Q.view(-1,1)
        
        Q_target = reward_batch + self._gamma * next_Q
        
        # Update policy
        self._optimizer.zero_grad()
        loss = self._loss_fn(Q, Q_target)
        loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self._Q.parameters(), 10)
        self._optimizer.step()
        
    def copy(self):
        """Copy method.
        
            This method copies the learning Q-function to the target Q-function.
            In the case of the function approximator, the state dict defined by
            PyTorch objects is used.
            See the generic method for further information.

            Args:
                None.

            Returns:
                Nothing.
        """
        self._Q_target.load_state_dict(self._Q.state_dict())

    def set_learning_parameters(self, num_episodes, optimizer, loss_fn, gamma, lr, lr_schedule=None, schedule_param={},
        epsilon=None, epsilon_0=1.0, explore_fraction=0.1, final_epsilon=0.02, **kwargs):
        """Set learning parameters.
        
            This method sets the learning parameters of the Q-Learning
            algortihm with function approximators.

            Args:
                num_episodes (int): Number of training episodes.
                optimizer (torch.optim): PyTorch optimizer instance, this
                    object defines how to optimize the function
                    approximator.
                loss_fn (torch.F): PyTorch loss function instance, this
                    object defines the loss function for the training.
                gamma (float): Discout factor, between zero and one.
                lr (float): Starting value for learning rate schedule.
                lr_schedule (torch.Schedule): Learning rate schedule object
                    from PyTorch.
                    Defines the schedule according to which the learning
                    rate is decayed, the parameters are passed using the
                    schedule_param dictionary. If None, the learning rate
                    is kept constant over the learning process.
                schedule_param (dict): Dictionary containing the parameters
                    of the learning rate schedule, those are passed to the
                    PyTorch schedule object.
                epsilon (Schedule): Exploration schedule object. Defines
                    the schedule according to which the exploration fraction
                    is decayed. If None, a linear schedule is defined
                    using the next few arguments.
                epsilon_0 (float): Starting value for exploration
                    schedule.
                explore_fraction (float): Fraction of episodes over which
                    the exploration fraction is decayed.
                final_epsilon (float): End value for the exploration
                    schedule.
                kwargs: Generic keyword arguments, which are passed to the
                    optimizer constructor.

            Note:
                The method implements a wrapper around the learning rate
                schedule to unify the interfaces of the schedules defined
                at the end of this module and the schedules implemented in
                PyTorch.
        """
        self._optimizer = optimizer(self._Q.parameters(), lr=lr, **kwargs)
        self._loss_fn = loss_fn()
        self._gamma = gamma
        if epsilon is None:
            self._epsilon = LinearSchedule(epsilon_0, explore_fraction, final_epsilon, num_episodes)
        else:
            assert Schedule in type(epsilon).__bases__
            self._epsilon = epsilon
        
        if lr_schedule is None:
            self._alpha = LinearSchedule(lr, 0.0, lr, 0.0)
        else:
            class Alpha(lr_schedule):
                def __init__(self, *args, **kwargs):
                    super(Alpha, self).__init__(*args, **kwargs)

                def __call__(self):
                    return self.optimizer.param_groups[0]['lr']
            
            self._alpha = Alpha(self._optimizer, **schedule_param)       

    def export(self):
        """Export method.
        
            This method exports the learning Q-function and is used to save
            a trained agent, i.e., the learned Q-function. In the case of a
            function approximator, the method returns the state dictionary
            of a PyTorch object.

            Args:
                None.

            Returns:
                Q (dict): State dictionary of a PyTorch function
                    approximator.
        """
        return self._Q.state_dict()

    def import_(self, agent):
        """Import method.
        
            This method imports a previously exported and/or trained Q
            approximator instance. For more information see the generic
            method.

            Args:
                agent (dict): State dictionary of a PyTorch function
                    approximator.
        """
        self._Q.load_state_dict(agent)
        self._Q.eval()


class BaseAgent(object):
    """Base class for Q-Learning algorithm.

        This class acts as a parent class to the two specific Q-Learning
        algorithms implemented below. The learning algorithm, called agent,
        uses the approximator classes defined in the classes above.

        This class defines all common methods for learning agents, most
        notably the _learn method, which is the heart of every agent.
        However, for any non Q-Learning agent, this class needs rewriting!

        Note:
            The class is implemented in a modular fashion to allow for the
            usage in distributed learning settings. Mearning, any number of
            approximators can be used within this class.
    """
    def __init__(self, env, seed=1, tboardpath='../tboardlogs/'):
        """Generic initialization method.

            This method initializes all common attributes of the specific
            agents.

            Args:
                env (gym.Env): Instance of an OpenAI gym environment. Possibly,
                    one of the models in models.py.
                seed (int): Random seed. Setting this allows reproducability.
                tboardpath (str): Path to a logging folder. TensorBoard logs
                    are produced by the agent, which are stored in this location.
                    This allows to use TensorBoard to view the learning process
                    live.
        """
        self._env = env
        self._logging_path = tboardpath
        self._done = False
        self._seed = seed
        self._Q = None
        self._parameters = {}
        self._successes = 0

        # set seeds
        logger.info("Using seed {}".format(seed))
        torch.manual_seed(seed)
        random.seed(seed)
        self.np_random, _ = seeding.np_random(seed)
        self._env.seed(seed)
        

    def _learn(self, num_episodes, safety_framework=None, memory_size=50000, batch_size=32, train_freq=1, target_update_freq=500,
        stop_tolerance=1e-4):
        """Common learn method.

            This method implements the base Q-Learning algorithm, which
            contains all common elements for all Q-Learning algorithms.
            Q-Learning algorithm: Off-policy TD control. Finds the optimal
            greedy policy, while following an epsilon-greedy policy.
        
            Args:
                num_episodes (int): Number of training episodes.
                safety_framework (SafetyFramework): Instance of a safety
                    framework. This should be a method, taking the state
                    and the input as arguments. This implementation allows
                    to use an arbitrary safety framework with the agent.
                memory_size (int): Size of the replay memory.
                batch_size (int): How many elements should be sampled
                    from the replay memory.
                train_freq (int): After how many episodes the Q-function
                    is updated, e.g. if 2, the Q-function is updated
                    in every second episode.
                target_update_freq (int): After how many episodes the
                    learning Q-function is copied to the target Q-function.
                stop_tolerance (float): If the average reward over 100
                    episodes doesn't change more than this relative quantity
                    the learning process is stopped.

            Note:
                This method writes all information to a TensorBoard file
                to log the whole learning process. Use TensorBoard to view
                the learning process live. Few information about the process
                are also written to the logging stream.

                If the learning process was successful the agent is labelled
                as trained.
        """
        reward_history = []
        memory = ReplayMemory(memory_size)
        total_steps = 1
           
        for episode in range(num_episodes):
            # Print out which episode we're on
            if (episode + 1) % 100 == 0:
                mean_reward = np.mean(reward_history[-100:-1])
                logger.info("Episode: {}/{}\tMean Reward: {}\tTotal Timesteps: {}".format(
                    episode + 1, num_episodes, np.round(mean_reward,3), total_steps))
                if episode > 200:
                    if (np.abs(mean_reward - reward_buffer) < mean_reward*stop_tolerance):
                        logger.info("Terminate learning... mean reward is constant")
                        break
                reward_buffer = mean_reward

            # Reset the environment and pick the first action
            state = self._env.reset()
            episode_reward = 0
            
            for s in count():
                if self.np_random.rand() < self._Q._epsilon():
                    action = self._env.action_space.sample()
                else:
                    action = self._Q.greedy_action(state)
                
                # safety framework
                if safety_framework is not None:
                    safe_action = safety_framework(state, action)
                    self._writer.add_scalar('data/action', action, total_steps)
                    self._writer.add_scalar('data/safe_action', safe_action, total_steps)
                    next_state, reward, done, _ = self._env.step(safe_action)
                    action = safe_action
                else:
                    next_state, reward, done, _ = self._env.step(action)

                memory.push(state, action, next_state, reward, done)
                
                if total_steps > batch_size and total_steps % train_freq == 0:
                    self._Q.update(memory.sample(batch_size))
                
                if total_steps % target_update_freq == 0:
                    self._Q.copy()
                
                # Update statistics
                episode_reward += reward
                total_steps += 1
                
                if done:
                    # successful episode
                    if reward == 1000: #TODO!!!!
                        self._successes += 1

                    reward_history.append(episode_reward)
                    self._Q._epsilon.step()
                    self._Q._alpha.step(episode_reward)
                    self._writer.add_scalar('data/episode_reward', episode_reward, episode)
                    self._writer.add_scalar('data/episode_length', s, episode)
                    self._writer.add_scalar('data/epsilon', self._Q._epsilon(), episode)
                    self._writer.add_scalar('data/successes', self._successes, episode)
                    self._writer.add_scalar('data/alpha', self._Q._alpha(), episode)
                    break
                else:
                    state = next_state
        
        self._writer.close()
        logger.info("---")
        logger.info("Complete!\tSuccesses: {}\tEpsilon: {}".format(self._successes, self._Q._epsilon()))
        logger.info("---")
        self._done = True

    def act(self, state):
        """Act method.

            This method evaluates the trained agent for the passed
            state and returns the action according to the learned
            policy.

            Args:
                state (np.array): Numpy array of the current state.
                    This can contain multiple states or just a single one.

            Returns:
                action (np.array): Action with the largest Q-value given
                    the current state. If multiple states are passed to
                    the method, multiple actions are returned.

            Raises:
                Warning: If the agent wasn't trained before calling this
                    method. In that case, the method returns empty.

            Note:
                If the largest Q-value is achieved by multiple actions, one
                of them is chosen randomly. See the documentation of the
                greedy_action method of the specific approximator for more
                information.
        """
        if not self._done:
            logger.warning("Agent is not trained, cannot act...")
            return
        return self._Q.greedy_action(state)

    def save(self):
        """Save method.

            This method saves the trained agent to the disk, to store a
            specific learning process. The learned policy as well as all
            learning parameters used, are stored to the disk. The agent
            is saved to the logging location, specified at initialization.

            Args:
                None.

            Raises:
                Warning: If the agent is not trained. The current state
                    of the agent is saved anyway.

            Note:
                This method calls the export method of the Q-function
                approximators. See that methods documentation for further
                details.
        """
        if not self._done:
            logger.warning("You are saving an untrained agent.")
        path = self._logging_path + "/parameters"
        logger.info("saving parameters to {}".format(path))
        pickle.dump(self._parameters, open(path, "wb"))
        path = self._logging_path + "/agent"
        logger.info("saving agent to {}".format(path))
        pickle.dump(self._Q.export(), open(path, "wb"))

    def load(self, logging_path):
        """Load method.

            This method loads a previously trained agent to an empty
            agent instance. Both, the learned policy and the learning
            parameters are loaded.

            Args:
                logging_path (str): Path to the agent you want to load.

            Raises:
                Warning: If the current instance already contains a
                    trained agent. In that case, the load process is
                    aborted. Generate a new empty agent instance to
                    load an agent.
                TypeError: If the agent you try to load is not of the
                    same type as the current agent, e.g. you are loading
                    an agent using a Q-table, while the current instance
                    uses function approximators.

            Note:
                This method calls the import_ method of the Q-function
                approximators. See that methods documentation for further
                details.
        """
        if self._done:
            logger.warning("Agent is already trained! Abort loading...")
            return
        self._logging_path = logging_path
        path = self._logging_path + "/parameters"
        logger.info("loading parameters from {}".format(path))
        self._parameters = pickle.load(open(path, "rb"))
        if self._Q.__class__ != self._parameters["class"]:
            logger.error("The agent you try to load is different from the current instance!")
            logger.error("{} vs. {}".format(self._parameters["class"], self._Q.__class__))
            self._parameters = {}
            return
        path = self._logging_path + "/agent"
        logger.info("loading agent from {}".format(path))
        agent = pickle.load(open(path, "rb"))
        self._Q.import_(agent)
        self._done = True

    def info(self):
        """Q-Learning info.
        
            This method prints the info and settings of the Q-Learning algorithm
            to the logging stream.
            This is the generic method, printing all common information. The
            specific information should be printed by the specific methods.

            Args:
                None.
        """
        logger.info("--- INFO ---")
        logger.info("random seed: {}".format(self._seed))
        logger.info("logging path: {}".format(self._logging_path))
        logger.info("---")
        logger.info("--- ENVIRONMENT ---")
        if 'env' in self._env.__dict__:
            logger.info(self._env.__dict__['env'])
        else:
            logger.info(self._env.__class__)
        if self._env.__doc__ is None:
            logger.info("No environment documentation found.")
        else:
            logger.info(self._env.__doc__)
        logger.info("---")


class DQL(BaseAgent):
    """Deep Q-Learning algorithm.

        This class implements a learning agent using PyTorch function
        approximators. Most commonly, those function approximators are
        neural networks, but in general they can by any function.

        See the generic base class for more informations.

        Note:
            The class uses the approximator class QFunction defined above.
    """
    def __init__(self, env, function_approximator, discretization=3, init_function=None, device=None, seed=1, tboardpath='../tboardlogs/'):
        """Initialization method.

            This method initializes the specific attributes of the agent
            using function approximators.

            Args:
                env (gym.Env): Instance of an OpenAI gym environment. Possibly,
                    one of the models in models.py.
                function_approximator (PyTorch object): Function approximator defined
                    by a PyTorch object instance.
                discretization (int or list): (List of) number of bins, to discretize
                    the action space into. Due to the usage of a continuous function
                    approximator the state doesn't need to be discretized.
                init_function (PyTorch object): Optional initialize function applied to
                    layered networks. PyTorch enables specific NN initialization with
                    function, this is enabled with this argument.
                device (str): Wether to use the CPU or GPU to train and evaluate PyTorch
                    objects. If None, the GPU is chosen if one is available.
                seed (int): Random seed. Setting this allows reproducability.
                tboardpath (str): Path to a logging folder. TensorBoard logs
                    are produced by the agent, which are stored in this location.
                    This allows to use TensorBoard to view the learning process
                    live.
        """
        super(DQL, self).__init__(env, seed, tboardpath)
        self._Q = QFunction(self._env, function_approximator, discretization, init_function, device, self.np_random)
        self._logging_path = '{0}q-learning_{1}'.format(self._logging_path, datetime.now().strftime('%Y%b%d_%H%M%S'))
        self._writer = SummaryWriter(self._logging_path)

    def learn(self, num_episodes, optimizer, loss_fn, safety_framework=None, memory_size=50000, gamma=0.99, lr=1e-3, batch_size=32,
        train_freq=1, target_update_freq=500, stop_tolerance=1e-4, **kwargs):
        """Deep Q-Learning train method.

            This method implements the training method for the agent
            using function approximators.
        
            Args:
                num_episodes (int): Number of training episodes.
                optimizer (torch.optim): PyTorch optimizer instance, this
                    object defines how to optimize the function
                    approximator.
                loss_fn (torch.F): PyTorch loss function instance, this
                    object defines the loss function for the training.
                safety_framework (SafetyFramework): Instance of a safety
                    framework. This should be a method, taking the state
                    and the input as arguments. This implementation allows
                    to use an arbitrary safety framework with the agent.
                memory_size (int): Size of the replay memory.
                gamma (float): Discout factor, between zero and one.
                lr (float): Starting value for learning rate schedule.
                batch_size (int): How many elements should be sampled
                    from the replay memory.
                train_freq (int): After how many episodes the Q-function
                    is updated, e.g. if 2, the Q-function is updated
                    in every second episode.
                target_update_freq (int): After how many episodes the
                    learning Q-function is copied to the target Q-function.
                stop_tolerance (float): If the average reward over 100
                    episodes doesn't change more than this relative quantity
                    the learning process is stopped.
                kwargs: Generic keyword arguments, passed to the set_
                    learning_parameters method.

            Raises:
                Warning: If the agent is already trained. This is done to
                    prevent overwriting a trained agent. The learning
                    process is aborted.
                RuntimeError: If the agent wasn't able to initialize a
                    Q-function approximator.

            Note:
                This method writes all information to a TensorBoard file
                to log the whole learning process. Use TensorBoard to view
                the learning process live. Few information about the process
                are also written to the logging stream.

                If the learning process was successful the agent is labelled
                as trained.

                For further possible arguments, which can be passed to the
                function to initialize the specific approximators have a
                look at the approximator classes' documentation.
        """
        if self._done:
            logger.warning("Agent is already trained! Abort learning...")
            return
        if self._Q is None:
            raise RuntimeError("No state-action value function specified!")

        self._parameters = {
        "class": self._Q.__class__,
        "num_episodes": num_episodes,
        "optimizer": optimizer,
        "loss_function": loss_fn,
        "safety_framework": safety_framework,
        "memory_size": memory_size,
        "gamma": gamma,
        "learning_rate": lr,
        "batch_size": batch_size,
        "train_freq": train_freq,
        "target_update_freq": target_update_freq,
        **kwargs
        }
        self._Q.set_learning_parameters(num_episodes, optimizer, loss_fn, gamma, lr, **kwargs)
        self._learn(num_episodes, safety_framework, memory_size, batch_size, train_freq, target_update_freq, stop_tolerance)

    def info(self):
        """Q-Learning info.
        
            This method prints the info and settings of the Q-Learning algorithm
            to the logging stream.

            Args:
                None.
        """
        super(DQL, self).info()
        logger.info("--- AGENT ---")
        logger.info("Using Q-learning with pytorch function approximator:")
        logger.info(self._Q._Q)
        logger.info("trained: {}".format(self._done))
        if self._done:
            logger.info("successful episodes: {}".format(self._successes))
            logger.info("learning parameters:")
            logger.info(self._parameters)
        logger.info("---")


class QL(BaseAgent):
    """Standard Q-Learning algorithm.

        This class implements a learning agent using Q-tables to
        approximate the Q-function.

        See the generic base class for more informations.

        Note:
            The class uses the approximator class QTable defined above.
    """
    def __init__(self, env, discretization=[100,3], custom_bounds={}, seed=1, tboardpath='../tboardlogs/'):
        """Initialization method.

            This method initializes the specific attributes of the agent
            using Q-tables.

            Args:
                env (gym.Env): Instance of an OpenAI gym environment. Possibly,
                    one of the models in models.py.
                discretization (list): List of the number of bins, to discretize
                    the state space and action space into.
                custom_bounds (dict): Dictionary containing OpenAI gym box objects.
                    Those box objects are either observation spaces or action spaces,
                    defining custom bounds for infinitely bounded states or actions.
                seed (int): Random seed. Setting this allows reproducability.
                tboardpath (str): Path to a logging folder. TensorBoard logs
                    are produced by the agent, which are stored in this location.
                    This allows to use TensorBoard to view the learning process
                    live.
        """
        super(QL, self).__init__(env, seed, tboardpath)
        self._Q = QTable(self._env, discretization, custom_bounds, self.np_random)
        self._logging_path = '{0}q-table_{1}'.format(self._logging_path, datetime.now().strftime('%Y%b%d_%H%M%S'))
        self._writer = SummaryWriter(self._logging_path)

    def learn(self, num_episodes, safety_framework=None, memory_size=50000, gamma=0.95, batch_size=32, train_freq=1,
        target_update_freq=500, stop_tolerance=1e-4, **kwargs):
        """Standard Q-Learning train method.

            This method implements the training method for the agent
            using Q-tables.
        
            Args:
                num_episodes (int): Number of training episodes.
                safety_framework (SafetyFramework): Instance of a safety
                    framework. This should be a method, taking the state
                    and the input as arguments. This implementation allows
                    to use an arbitrary safety framework with the agent.
                memory_size (int): Size of the replay memory.
                gamma (float): Discout factor, between zero and one.
                batch_size (int): How many elements should be sampled
                    from the replay memory.
                train_freq (int): After how many episodes the Q-function
                    is updated, e.g. if 2, the Q-function is updated
                    in every second episode.
                target_update_freq (int): After how many episodes the
                    learning Q-function is copied to the target Q-function.
                stop_tolerance (float): If the average reward over 100
                    episodes doesn't change more than this relative quantity
                    the learning process is stopped.
                kwargs: Generic keyword arguments, passed to the set_
                    learning_parameters method.

            Raises:
                Warning: If the agent is already trained. This is done to
                    prevent overwriting a trained agent. The learning
                    process is aborted.
                RuntimeError: If the agent wasn't able to initialize a
                    Q-function approximator.

            Note:
                This method writes all information to a TensorBoard file
                to log the whole learning process. Use TensorBoard to view
                the learning process live. Few information about the process
                are also written to the logging stream.

                If the learning process was successful the agent is labelled
                as trained.

                For further possible arguments, which can be passed to the
                function to initialize the specific approximators have a
                look at the approximator classes' documentation.
        """
        if self._done:
            logger.warning("Agent is already trained! Abort learning...")
            return
        if self._Q is None:
            raise RuntimeError("No state-action value function specified!")

        self._parameters = {
        "class": self._Q.__class__,
        "num_episodes": num_episodes,
        "safety_framework": safety_framework,
        "memory_size": memory_size,
        "gamma": gamma,
        "batch_size": batch_size,
        "train_freq": train_freq,
        "target_update_freq": target_update_freq,
        **kwargs
        }
        self._Q.set_learning_parameters(gamma, num_episodes, **kwargs)
        self._learn(num_episodes, safety_framework, memory_size, batch_size, train_freq, target_update_freq, stop_tolerance)

    def info(self):
        """Q-Learning info.
        
            This method prints the info and settings of the Q-Learning algorithm
            to the logging stream.

            Args:
                None.
        """
        super(QL, self).info()
        logger.info("--- AGENT ---")
        logger.info("Using Q-learning with a Q-table")
        logger.info("Q-table shape: {}".format(self._Q._Q.shape))
        logger.info("trained: {}".format(self._done))
        if self._done:
            logger.info("successful episodes: {}".format(self._successes))
            logger.info("learning parameters:")
            logger.info(self._parameters)
        logger.info("---")

"""Definition of the Transition structure used in the ReplayMemory"""
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    """Replay Buffer.

        This class implements a replay buffer commonly encountered
        in reinforcement learning methods. It is required in deep
        Q-learning to stabilize the learning algorithm, see ..
    """
    def __init__(self, capacity):
        """Initialize the replay buffer.

            Initializes the capacity and allocates an empty list for the
            buffer. The replay buffer is wrapped, i.e., if the memory is
            full, the first entry is overwritten.

            Args:
                capacity (int): Capacity/Size of the memory.
        """
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Add an element to the memory.

            This method appends a new element at the end of the memory,
            unless the memory is full, then is starts from the beginning
            again.

            Args:
                *args (): Sorted arguments of a Transition (as defined above).
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample the buffer randomly.

            This method returns a random sample of size batch_size from
            the memory.

            Args:
                batch_size (int): Size of the batch to be sampled.

            Returns:
                batch (list): A batch of randomly sampled elements
                    from the memory (an element is a Transition structure).
        """
        batch = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*batch))
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Lenght attribute definition.

            This method defines the length of the ReplayMemory object to
            be the current memory size.

            Returns:
                size (int): Current size of the memory. This is not
                    necessarily equal to the capacity, e.g. if the memory
                    is not full.

            Note:
                If the len() method is called with an object, this attributed
                is checked and returned. This attributed is not defined for
                objects by default.
        """
        return len(self.memory)

class Idx(object):
    """Indexing class.

        This class implements an indexing object, which allows
        to compute the index of a state or action, given a
        specific discretization of an observation space or
        action space.
    """
    def __init__(self, box, discretization):
        """Initalization method.

            This method initilazes the indexing class.

            Args:
                box (gym.Box): OpenAI gym box object, this
                    object defines the bounds of the state
                    or action space.
                discretization (list): Number of bin, into
                    which the state or action space dimensions
                    should be discretized.

            Note:
                The discretization in different dimensions
                doesn't need to be the same.
        """
        if type(box) is Box:
            self.low = box.low
            width = box.high - box.low
            self.width = width
            if np.any(np.abs(width) > 1e6):
                logger.warning("The environment may has infinite bounds! Consider manually capping them.")
            self.n = len(discretization)
            self.d = np.array(discretization)
            self.dtype = box.dtype
        elif type(box) is Discrete:
            self.d = None
            self.n = 1
                
    def __call__(self, state):
        """Call interface.

            This method allows to get the index of a state or
            action by calling the indexing instance.

            Args:
                state (float or list): State(s) for which the
                    index(es) should be computed.

            Returns:
                index (float or list): Computed index(es) of
                    passed state(s).
        """
        if self.d is not None:
            if len(state.shape) == 1:
                state = state[newaxis]
            assert self.n == state.shape[1]
            idx = (state - self.low)/self.width * (self.d - 1)
            idx = np.round(idx).astype(np.int)
            idx = np.hsplit(idx, self.n)
            for n in range(self.n):
                idx[n] = idx[n].clip(0,self.d[n]-1).squeeze().tolist()
            return idx
        else:
            if type(state) is int:
                return [state]
            else:
                return [state.squeeze().tolist()]

    def reverse(self, idx):
        """Reverse of standard call.

            This method computes the state or action, given
            the index.

            Args:
                idx (float or list): Index(es) for which the
                    state(s) should be computed.

            Returns:
                index (float or list): Computed state(s) of
                    passed index(es).
        """
        if self.d is None:
            return idx
        else:
            state = idx/(self.d - 1) * self.width + self.low
            return state.astype(self.dtype)

class Schedule(object):
    """Base class for learning schedules.

        This class acts as a parent class to any specific learning
        schedule, e.g. the two schedules implemented below. It
        implements the methods shared by all schedule classes and
        defines the basic interface.

        A schedule is a time sequence of a given quantitiy, i.e.,
        a function of time. An example is the exploration parameter
        epsilon, which is commonly reduced over time in most
        reinforcement learning methods, in order to exploit the learned
        information and stop learning.
    """
    def __init__(self, base):
        """Initialize schedule.

            This method defines the actual learning schedule. For a
            specific schedule, the _schedule attribute needs to be
            defined as a lambda function.

            Args:
                base (float): Start value of the schedule.

            Note:
                This is the only method, which should be changed to
                define a specific schedule.
        """
        self._i = 0
        self._value = base
        
    def __call__(self):
        """Call interface.

            This method allows to get the current value of the schedule
            by calling the class instance directly.

            Args:
                None.

            Returns:
                value (float): Current value of the schedule.
        """
        return self._value

    def step(self, *args):
        """Step method.

            Increments the counter of the schedule by one and
            computes the next value of the schedule.

            Args:
                args: Generic arguments passed to the _schedule
                    function.

            Returns:
                value (float): Current value of the schedule.
        """
        self._value = self._schedule(self._i, *args)
        self._i += 1
        return self._value

class LinearSchedule(Schedule):
    """Linear Schedule.

        Linearly decaying schedule, starting from a start value and
        decaying to an end value, given a decay rate defined by the
        fraction of a total number of iterations.
    """
    def __init__(self, base, fraction, end_value, total_iterations):
        """Initialize linear schedule.

            Args:
                base (float): Start value of the schedule.
                fraction (float): Fraction of the total number of iterations
                    over which the schedule decays the value from base to
                    end_value.
                end_value (float): End value of the schedule, the schedule
                    doesn't decay the value lower than this.
                total_iterations (float): Total number of iterations.

            Note:
                Also includes the constant schedule case, i.e. start value
                and end value are the same, or the fraction is zero.
        """
        super(LinearSchedule, self).__init__(base)
        if fraction == 0 or base == end_value:
            self._schedule = lambda t: base
        else:
            self._schedule = lambda t: base + min(float(t) / int(fraction * total_iterations), 1.0) * (end_value - base)

class DecaySchedule(Schedule):
    """Linear Schedule.

        Linearly decaying schedule, starting from a start value and
        decaying to an end value, given a decay rate defined by the
        fraction of a total number of iterations.
    """
    def __init__(self, base, decay):
        """Initialize linear schedule.

            Args:
                base (float): Start value of the schedule.
                fraction (float): Fraction of the total number of iterations
                    over which the schedule decays the value from base to
                    end_value.
                end_value (float): End value of the schedule, the schedule
                    doesn't decay the value lower than this.
                total_iterations (float): Total number of iterations.

            Note:
                Also includes the constant schedule case, i.e. start value
                and end value are the same, or the fraction is zero.
        """
        super(DecaySchedule, self).__init__(base)
        self._schedule = lambda t: base * (1.0 / (1.0 + decay * t))
        