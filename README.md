# ICRA2020_RM_decision-making

## 1 Preface
Because of the 2020 epidemic, school management does not allow us to go back to school, so most of our work is online.

### 1.1 Hardware
We test the inference speeds of Deep-FSMs on an NVIDIA Jetson TX2, a GPU platform targets at power-constrained mobile applications.

## 2 Deep-FSM
 We separate the perception, decision-making, and control parts of robots. We model the perception part and put it into our deep finite state machine (Deep-FSM). Each state of the Deep-FSM corresponds to an atomic robot behavior composed of underlying control algorithms. At the same time, we design several training stages to alleviate the state oscillation of Deep-FSMs. And for a multi-agent environment, agents can use the hierarchical deep finite state machine (HDFSM) to make decisions.

### 2.1 Simulation design
To study robot decision-making more conveniently, we set up simulation environments using Gazebo and OpenAI gym. As shown in Fig.1 and Fig.2, our simulation environments include robots, obstacle, Buff/Debuff zone, the arena map, and other basic elements. These elements are designed in strict accordance with the rules of the competition. The robots contain the basic commands of forwarding, backward, left and right rotation and shooting. And we can set the basic attributes of each robot, such as speed, blood volume, size, and bullet speed. Buff/Debuff Zones include a Restoration Zone, a Projectile Supplier Zone, a No Shooting Zone, and a No Moving Zone. In simulations, the maps have the same proportion as the real world, and the obstacles are placed in the same proportion. In the real environment, because of the limited computing power and sensor precision, we cannot always get the correct and complete information. We integrate the sensor information from the real environment, which is called perceptual modeling. And in simulations, we use the information in the same format of the perception model as the partial observations of the environment ![](http://latex.codecogs.com/svg.latex?x^e).

![](https://github.com/gongpx20069/ICRA2020_RM_decision-making/blob/master/img/arena.png)

![](https://github.com/gongpx20069/ICRA2020_RM_decision-making/blob/master/img/2D2v2.png)

One of the most important things in simulation design is the design of rewards. In simulations, the interactions between environments and robots are easy to execute. And in the competition, we can also get feedback from the environment to robots through the official referee system. Because it is unrealistic to train many times in the real environment, we store the interactive data in every training process, and extract small batch of training, so that the historical data can be used effectively. 

Behaviors | Rewards 
:-: | :-: 
enter own Buff Zones | +10 
enter enemy Buff Zones | -10 
enter Debuff Zones | -10 
hit the enemy | +10 
hit the teammate | -10 
shoot bullets | -1 

### 2.2 Deep-FSM
Our deep finite state machine (Deep-FSM) requires two inputs, the partial observations of the environment ![](http://latex.codecogs.com/svg.latex?x^e), and the current state of the Deep-FSM ![](http://latex.codecogs.com/svg.latex?x^s). Whether it is in the competition or simulations, our observations are an array composed of 28 elements, including the locations of our side and the enemy, the remaining blood volumes, the numbers of bullets, the activation status and locations of six Buff/Debuff Zones. In the simulation environment built by OpenAI gym, we can easily get the above observations ![](http://latex.codecogs.com/svg.latex?x^e). But in the real world, we need to build a perception model.

![](https://github.com/gongpx20069/ICRA2020_RM_decision-making/blob/master/img/DeepFSM.jpg)

when we have two enemies, we can use the first layer of the HDFSM to select the target and integrate the corresponding observation ![](http://latex.codecogs.com/svg.latex?x^e) to the next layer of the HDFSM to make the final behavior decision. The method is called the hierarchical deep finite state machine (HDFSM)

![](https://github.com/gongpx20069/ICRA2020_RM_decision-making/blob/master/img/HDFSM.jpg)

### 2.3 Robot behaviors
The state ![](http://latex.codecogs.com/svg.latex?x^s) of the Deep-FSM is an array of five elements, respectively representing five robot behaviors of "shooting", "chasing", "escaping", "adding blood" and "adding bullets". The bottom implementations of robot behaviors in the competition and simulations are slightly different, which does not affect the training and decision-making of the Deep-FSM, though.
We can design different robot behaviors. After careful screening, we found five indispensable robot behaviors, named atomic robot behaviors. The five atomic robot behaviors are: "shooting", "chasing", "escaping", "adding blood" and "adding bullets", corresponding to five states of the Deep-FSM.

    - Shooting state: Shooting state is mainly realized through robot control algorithm of aiming shooting 
    - Chasing state: In the chasing state, our robot will use the path planning algorithm to generate a route whose destination is the location of the enemy. 
    - Escaping state: Take the center point of the map as the origin, the wide edge of the map as the X-axis, and the heigh edge of the map as the Y-axis to establish the coordinate system. In this coordinate system, we divide the map into four quadrants. In the escape state, our robot will use the path planning algorithm to generate a path whose destination is in the opposite quadrant of the enemy's location and there is an obstacle or more between the enemy and the destination.  
    - Adding blood state: In the adding blood state, our robot will use the path planning algorithm to plan a path whose destination is Restoration Zone, which will help our robot recover 200 health points.
    - Adding bullets state: In the adding bullets state, our robot will use the path planning algorithm to plan a path whose destination is Projectile Supplier Zone, which will equip our robot with 100 bullets.

### 2.4 PPO
Proximal Policy Optimization Algorithms is a stochastic reward based policy search method which has been shown to stably learn policies for continuous observation and action spaces. The architecture involves an actor-critic neural network pair, in which actor observes to provide actionable actions, while critic provides estimates of expected discounted long-term reward for a given strategy.

## 3 Behavior tree
we also design a behavior tree model whose input is ![](http://latex.codecogs.com/svg.latex?x^e) and output is an atomic robot behavior. We wrap the output as ![](http://latex.codecogs.com/svg.latex?x^e), which has the same size as the state of the Deep-FSM ![](http://latex.codecogs.com/svg.latex?x^s)

![](https://github.com/gongpx20069/ICRA2020_RM_decision-making/blob/master/img/BT.jpg)

## 4 Code interpretation
In this section, we will introduce our code in detail.
### 4.1 map.py
This script defines the simulation map of thThis script defines the simulation map of the 2020 ICRA competition and the rules for judging whether the robot touches the edge. At the same time, a cost map is provided for robots. 
We will call the map in code 4.2 and 4.3.

### 4.2 PRM_AStar.py
In this part of the code, we mainly implement the robot path planning algorithm astar algorithm, and use PRM to reduce the calculation workload, so that the astar algorithm can be executed in real time. It can quickly provide a planned route for the robot.
We will call the PRM_AStar in code 4.4.

### 4.3 RoboMaster.py/RoboMaster1v2.py/RoboMaster2v2.py
In this part, we use the gym framework provided by openAI to implement the reinforcement learning simulation of ICRA competition. Different observations and rewards of 1v1, 1V2 and 2v2 are defined in different scripts.

### 4.4 DFSM.py
In this part, we implement the behavior of robots in different states in deepfsm, and the corresponding underlying implementation methods (path planning, shooting etc.).

### 4.5 DFSM_PPO_RM.py
In this script, we implement PPO algorithm for deep reinforcement learning of Deep-fsm model, including different numbers of robot confrontation.

### 4.6 DFSM_show/DFSM_show_1v2.py/DFSM_show_2v2.py/
For the trained model, we can use these scripts to demonstrate and calculate the winning rate of different implementation methods.
