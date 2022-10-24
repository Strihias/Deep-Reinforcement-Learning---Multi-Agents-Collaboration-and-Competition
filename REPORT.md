For this project we have used a slight modification of [DDPG Algorithm](https://arxiv.org/abs/1509.02971) in order to train the multi agents (in our case two). 

## Algorithm and Model Architecture

The architecture is comprised of two neural networks one for the Actor and one for the Critic as described below:

### Actor NN

* hidden layer: (input, 256) - ReLU
* hidden layer: (256, 128) - ReLU
* output layer: (128, 4) - tanH

### Critic NN

* hidden layer: (input, 256) - ReLU
* hidden layer: (256 + action size, 128) - ReLU
* output layer: (128, 1) - Linear

Using the Actor-Critic paradigm the training loop is composed out of two steps: acting and learning. 
In the acting step, the agent passes the state vector through the Actor network and takes the action which is the output of the network.
In the learning step, the Critic network is used as a feedback to the Actor network to change its weights such that the estimated value of the input state is maximized.

One addition compared to the originally proposed implementation of DDPG Algorithm was that we introduced local and target network weights both for the actor and the critic at initialization of the agent. 

Every agent receives its own (local) observation, but both agents share one actor network and one replay buffer. Both agents collaboratively learn to play tennis by competing in the game at the same time.

## Hyperparameters
* BUFFER_SIZE = int(1e5)  # replay buffer size
* BATCH_SIZE = 128         # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR_ACTOR = 1e-4         # learning rate of the actor
* LR_CRITIC = 1e-4        # learning rate of the critic
* WEIGHT_DECAY = 0      # L2 weight decay

## Results
Using the aforementioned architecture and hyperparameters we were able to solve the environment in 4002 episodes getting an average score of 0.5

We can see the performance of the model in the following diagram: 

![alt text](https://github.com/Strihias/Deep-Reinforcement-Learning---Multi-Agents-Collaboration-and-Competition/blob/main/diagram.jpg "Performance Diagram")

## Future Work
* Experiment more on Hyperparameter tuning
* Try more algorithms 
