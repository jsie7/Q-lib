## Q-lib
The Q-lib module implements the standard Q-Learning method and an extension to function approximators, using the [PyTorch](https://pytorch.org/) framework. The learning algorithms are implemented in a way, which allows for easy usage in distributed settings. However, distributed learning algorithms have to be implemented in future work and are not provided in this module.

Usage Q-Learning:
```python
from q-lib import QL
# create agent
agent = QL(env, discretization, *args, **kwargs)
# train the agent
agent.learn(num_episode, gamma, alpha, epsilon, *args, **kwargs)
# save the trained agent
agent.save()
```

Usage Q-Learning with function approximators:
```python
from q-lib import DQL
# create agent
agent = DQL(env, function_approximator, *args, **kwargs)
# train the agent
agent.learn(num_episode, optimizer, loss_function, memory_size, gamma, lr, *args, **kwargs)
# save the trained agent
agent.save()
```

where:
- *env* [object], is a gym environment object.
- *discretization* [list], is a list containing the number of discrete states and actions. The environment's states (called observations) and actions are subsequently discretized.
- *function_approximator* [object], is a PyTorch object of the function approximator used for the Q-function.
- *num_episode* [int], is the number of episodes for which the algorithm should learn.
- *gamma* [float], is the discout factor (between 0 and 1).
- *alpha* [float or Schedule], is the learning rate for the Q-Learning algorithm, can be constant or a Schedule object.
- *epsilon* [float or Schedule], is the exploration fraction, can be constant or a Schedule object.
- *optimizer* [pytorch.optim], is a PyTorch Optimizer object.
- *loss_function* [pytorch.nn], is a PyTorch Function object.
- *memory_size* [int], is the size of the replay memory.
- *lr* [float], is the learning rate for the Deep Q-Learning algorithm, has to be constant and between 0 and 1.
- *args* and *kwargs*, are further arguments passed to the respective method. See documentation for further info.

The Q-Learning module also contains the implementation of a replay memory and some implementations of schedules. The schedules are imported for example as:
```python
from q-lib import LinearSchedule
```
