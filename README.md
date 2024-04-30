# MAB_Electronic-Commerce-Models
This Python module implements a Planner class for a Multi-Armed Bandit simulation, managing arm selection via UCB and tracking statistics on arm performance and user interactions.

my idea implemen-ng the planner class in big picture is using maximum likelihood estmation
and UCB algorithm we learned at class, with more details:
In __init___ method I used parameters to calculate the statistics of the arms and the user.
In choose_arm method , first I used the given data in order to adapt to the simulation and gain maximum reward, in the first 90% of the iteration I used MLE, then I used UCB after I gained enough information regarding the ERM matrix to decide which arm gains the most reward , I also considered the user distribution like user with high distribution to be picked , we need to keep their most rewarding arm active for more time .
