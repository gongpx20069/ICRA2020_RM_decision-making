# ICRA2020_RM_decision-making

## 1 Preface
Because of the 2020 epidemic, school management does not allow us to go back to school, so most of our work is online.

## Hardware
We test the inference speeds of Deep-FSMs on an NVIDIA Jetson TX2, a GPU platform targets at power-constrained mobile applications.

## 2 Deep-FSM
 We separate the perception, decision-making, and control parts of robots. We model the perception part and put it into our deep finite state machine (Deep-FSM). Each state of the Deep-FSM corresponds to an atomic robot behavior composed of underlying control algorithms. At the same time, we design several training stages to alleviate the state oscillation of Deep-FSMs. And for a multi-agent environment, agents can use the hierarchical deep finite state machine (HDFSM) to make decisions.

### 2.1 Simulation design
To study robot decision-making more conveniently, we set up simulation environments using Gazebo and OpenAI gym. As shown in Fig.1 and Fig.2, our simulation environments include robots, obstacle, Buff/Debuff zone, the arena map, and other basic elements. These elements are designed in strict accordance with the rules of the competition. The robots contain the basic commands of forwarding, backward, left and right rotation and shooting. And we can set the basic attributes of each robot, such as speed, blood volume, size, and bullet speed. Buff/Debuff Zones include a Restoration Zone, a Projectile Supplier Zone, a No Shooting Zone, and a No Moving Zone. In simulations, the maps have the same proportion as the real world, and the obstacles are placed in the same proportion. In the real environment, because of the limited computing power and sensor precision, we cannot always get the correct and complete information. We integrate the sensor information from the real environment, which is called perceptual modeling. And in simulations, we use the information in the same format of the perception model as the partial observations of the environment $x^e$.

One of the most important things in simulation design is the design of rewards. In simulations, the interactions between environments and robots are easy to execute. And in the competition, we can also get feedback from the environment to robots through the official referee system. Because it is unrealistic to train many times in the real environment, we store the interactive data in every training process, and extract small batch of training, so that the historical data can be used effectively. 

\begin{table}[htbp]
    \centering
    \begin{tabular}{lp{3cm}p{3cm}}
         \hline
         \textbf{Behaviors} & \textbf{Rewards} \\
         \hline
         enter own Buff Zones & +10 \\
         enter enemy Buff Zones & -10 \\
         enter Debuff Zones & -10 \\
         hit the enemy & +10 \\
         hit the teammate & -10 \\
         shoot bullets & -1 \\
        %  higher health points after settlement & +100 \\
        %  equal health points after settlement & -10 \\
        %  lower health points after settlement & -100 \\
         \hline
    \end{tabular}
    \caption{Environment rewards for specific robot behaviors}
    \label{tab:reward}
\end{table}

### 2.2 Deep-FSM
Our deep finite state machine (Deep-FSM) requires two inputs, the partial observations of the environment $ x^e $, and the current state of the Deep-FSM $ x^s $. Whether it is in the competition or simulations, our observations are an array composed of 28 elements, including the locations of our side and the enemy, the remaining blood volumes, the numbers of bullets, the activation status and locations of six Buff/Debuff Zones. In the simulation environment built by OpenAI gym, we can easily get the above observations $ x^e $. But in the real world, we need to build a perception model.

when we have two enemies, we can use the first layer of the HDFSM to select the target and integrate the corresponding observation $x^e$ to the next layer of the HDFSM to make the final behavior decision. The method is called the hierarchical deep finite state machine (HDFSM)

### 2.3 Robot behaviors
The state $x^s$ of the Deep-FSM is an array of five elements, respectively representing five robot behaviors of "shooting", "chasing", "escaping", "adding blood" and "adding bullets". The bottom implementations of robot behaviors in the competition and simulations are slightly different, which does not affect the training and decision-making of the Deep-FSM, though.
We can design different robot behaviors. After careful screening, we found five indispensable robot behaviors, named atomic robot behaviors. The five atomic robot behaviors are: "shooting", "chasing", "escaping", "adding blood" and "adding bullets", corresponding to five states of the Deep-FSM.

    - Shooting state: Shooting state is mainly realized through robot control algorithm of aiming shooting 
    - Chasing state: In the chasing state, our robot will use the path planning algorithm to generate a route whose destination is the location of the enemy. 
    - Escaping state: Take the center point of the map as the origin, the wide edge of the map as the X-axis, and the heigh edge of the map as the Y-axis to establish the coordinate system. In this coordinate system, we divide the map into four quadrants. In the escape state, our robot will use the path planning algorithm to generate a path whose destination is in the opposite quadrant of the enemy's location and there is an obstacle or more between the enemy and the destination.  
    - Adding blood state: In the adding blood state, our robot will use the path planning algorithm to plan a path whose destination is Restoration Zone, which will help our robot recover 200 health points.
    - Adding bullets state: In the adding bullets state, our robot will use the path planning algorithm to plan a path whose destination is Projectile Supplier Zone, which will equip our robot with 100 bullets.

### 2.4 PPO
Proximal Policy Optimization Algorithms is a stochastic reward based policy search method which has been shown to stably learn policies for continuous observation and action spaces. The architecture involves an actor-critic neural network pair, in which actor observes to provide actionable actions, while critic provides estimates of expected discounted long-term reward for a given strategy.

## 3 Behavior tree
we also design a behavior tree model whose input is $x^e$ and output is an atomic robot behavior. We wrap the output as $BTree(x^e)$, which has the same size as the state of the Deep-FSM $x^s$.

