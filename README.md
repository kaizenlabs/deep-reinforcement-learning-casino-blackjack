# Double DQN with Deep Neural Networks For Casino Blackjack

[![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1ZzWGdfUdZmSxfi6PagIXg_U8OZqY3X0n?usp=sharing)

Double Q-Learning Implementation with Deep Neural Network (Double DQN) is proposed in H. van Hasselt, 2016 of Google Deep Mind, inspired by Double Q-Learning, Double DQN uses two different Deep Neural Networks, Deep Q Network (DQN) and Target Network.

https://arxiv.org/pdf/1509.06461.pdf

In this notebook, I aimed to use reinforcement learning and implement Google Deepmind's Double DQN to approximate a strategy for playing casino blackjack. 

The strategy approximates two separate strategies — one strategy if the player is holding a playable ace, and if they are not. 

The strategies employed by the RL agent are shown in a cross-matrix, between the dealer's shown card and the player's card sum, denoting whether the player should hit or stand. 

Disclaimer: Don't use this for gambling, I only trained it on 10,000 episodes. 

## Background on Q-Learning


In fundamental Q-learning, the Agent's optimal strategy involves selecting the most favorable action in any given state, based on the assumption that this action possesses the highest expected/estimated Q-value. However, since the Agent lacks knowledge about the environment initially, it must initially estimate Q(s, a) and update these estimates iteratively. These Q-values are prone to considerable noise, making it uncertain whether the action with the maximum expected/estimated Q-value is genuinely the best choice.

Regrettably, the action deemed optimal often yields smaller Q-values compared to non-optimal actions in many instances. In accordance with the basic Q-learning policy, the Agent tends to opt for non-optimal actions in a given state simply because they have higher Q-values. This issue is known as the overestimation of action value (Q-value).

When this problem arises, the noise from the estimated Q-value introduces substantial positive biases in the updating procedure. Consequently, the learning process becomes intricate and disorderly.

The idea of double Q-learning is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation. In the vanilla implementation, the action selection and action evaluation are coupled. We use the target-network to select the action and estimate the quality of the action at the same time.

Double Q-learning tries to decouple these procedures from one another by separting the action selection and action evaluation in two different deep neural networks (see note cell #7 - DQN Class).

### Colab Usage

Shift Enter to run each cell individually or make a copy of the notebook in the File Menu -> click Runtime, and then Run All.