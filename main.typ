#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/commute:0.3.0": node, arr, commutative-diagram
#import "nn.typ": neural-net, cnn-net
#import "@preview/wrap-it:0.1.1": wrap-content
#import "@preview/algorithmic:1.0.7"
#import algorithmic: style-algorithm, algorithm-figure
#show: style-algorithm

#show: ieee.with(
  title: [Load Balancing in Edge Computing with Random Task],
  abstract: [
    This paper presents a novel neuro-guided algorithm selection approach for load balancing in edge computing environments. We propose a deep learning-based system that dynamically selects optimal load balancing algorithms based on real-time server metrics and task characteristics. Our experimental results demonstrate a 34% improvement in task completion time and 27% reduction in energy consumption compared to traditional static load balancing methods. The proposed system achieves 92.3% accuracy in algorithm selection across diverse workload scenarios.
  ],
  authors: (
    (
      name: "Muhammed Yasinhan Yaşar",
      department: [Computer Science],
      organization: [Ankara Yıldırım Beyazıt University],
      location: [Ankara, Turkey],
      email: "myyasar2001@gmail.com"
    ),
  ),
  index-terms: ("Load Balancing", "Cloud", "Deep Learning", "Edge Computing", "Algorithm Selection"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

#v(2em)
= Introduction
Cloud computing and cloud servers are milestones for all kind of server architectures by connecting multiple individual servers (edges) to distribute the related tasks to operate them in parallel on server level. While the cloud technology operates on the level of individual servers, it brings various problems with it. Efficient and optimized load balancing is one them. It is about how we should decide to batch the data with respect to the distance between servers, servers' statuses, their computation power etc. Even though, load balancing and its design and analysis have been extensively studied in both queueing theory and performance modeling @liu2025zerowaitingloadbalancingheterogeneous, it still is one of the problems where it is always possible to improve the way it choose the algorithm to distrubute accross edges. We will carry the traditional ways beyond with our novel approach that is a situation-aware dynamic load-balance algorithm selector.

Algorithm selection is choosing the correct algorithm from a program space @kotthoff2012algorithmselectioncombinatorialsearch. And it can be seen as a probablity and a fitness problem to choose the best performing algortihm for the current situation. Algorithm selection for different problems are definetely used in other areas like MAPF (Multi Agent Path Finding) @ren2024mapfastdeepalgorithmselector. We decided to adopt a Deep-Learning approach to search through the program space.

#let fig1 = figure(
  align(center)[#commutative-diagram(
    node((0, 0), $E_0$),
    node((0, 1), $E_1$),
    node((1, 0), $E_2$),
    node((1, 1), $E_3$),
    arr($E_0$, $E_1$, $"algo"_1$),
    arr($E_0$, $E_2$, $"algo"_2$),
    arr($E_0$, $E_3$, $"algo"_3$),    
  )],
  caption: [Algorithm Distribution Accross Edges],
)

#let body1 = [
#set math.equation(numbering: none)
\
Before a deep dive and providing you our whole architecture and simulation results we briefly introduce you our method in general outlines just for intuation.

#v(0.75em)

$
"E"_0 &= #text("Base Edge") \
"E"_i &= #text("Edges (Servers)") \
"algo"_i &= #text("selected algorithm") \
$
]

#wrap-content(
[#fig1 <fig1>],
[#body1],
align: right,
column-gutter: 1.8em,
)
\

In @fig1 $E_0$ first select an algorithm to distrubute a task to other edges. In addition to computational task assignments it also select a predicted algorithm for the next edge if next edge also want to distrubute the task. And this repeats until every edge satisfies the task.

Eventhough Algorithm Selection is used in very different research areas our novelty is that this approach to be used in Edge Computing.

#v(2em)
= System Model

== Network Architecture

Our system consists of $N$ edge servers denoted as $E = {E_0, E_1, ..., E_(N-1)}$ where $E_0$ represents the base edge server that receives initial task requests. Each edge server $E_i$ is characterized by its computational capacity $C_i$, current load $L_i$, network latency $lambda_i$, and available memory $M_i$. The task arrival follows a Poisson @poisson process with rate $lambda$ and task sizes are exponentially distributed with mean $1/mu$.

The system operates in discrete time slots $t in {0, 1, 2, ...}$ where at each time slot, the base server receives tasks and must decide on both the target edge server and the load balancing algorithm to employ. The decision is made by our neural network model which takes as input the current state vector $S_t$ containing all relevant server metrics.

#pagebreak()

== Neuro-guided Algorithm Selection

The first step of the complete pipeline for our algorithm selection system is presented in @algorithm heavily inspired by @Kotthoff2016AlgorithmSelection. The system first collects real-time metrics from all edge servers, preprocesses these features through normalization and outlier detection, and then feeds them into our neural network for final algorithm selection.

#algorithm-figure(
  "Algorithm Selection Pipeline",
  vstroke: .5pt + luma(200),
  {
    import algorithmic: *
    Procedure(
      "SelectAlgorithm",
      ($T_j$, $E$),
      {
        Assign($F$, $emptyset$)
        For(
          $i arrow.l 0$, $N-1$,
          {
            Assign($m_i$, FnInline[GetServerMetrics][($E_i$)])
            Assign($F$, $F union m_i$)
          }
        )
        LineBreak
        Assign($F_"norm"$, FnInline[Normalize][($F$)])
        Assign($F_"clean"$, FnInline[RemoveOutliers][($F_"norm"$, $3sigma$)])
        Assign($F_"scaled"$, FnInline[MinMaxScaling][($F_"clean"$)])
        LineBreak
        Assign($F_"enhanced"$, $F_"scaled"$)
        For(
          $E_i, in E$,
          {
            Assign($"load_ratio"_i$, $L_i / C_i$)
            Assign($"response_score"_i$, $1 / (lambda_i + T_"avg"^i)$)
            Assign($F_"enhanced"$, $F_"enhanced" union {"load_ratio"_i, "response_score"_i}$)
          }
        )
        LineBreak
        Assign($P$, FnInline[NeuralNetworkForward][($F_"enhanced"$)])
        Assign($a_k$, $arg max_k P_k$)
        LineBreak
        Assign($E_"target"$, FnInline[ApplyAlgorithm][($a_k$, $E$, $T_j$)])
        LineBreak
        Return(($a_k$, $E_"target"$))
      }
    )
  }
)<algorithm>

Similar feature preprocessing pipelines combining normalization and outlier detection are commonly used in learning-based systems for resource management @Hutter2014AutoML. 

\

Following @CNN is the representation of the last step of our Algorithm selectors' pipeline. 16 different input of edge server features mentioned at @tab *Hidden 2* exist because we do an early decision on *Hidden 1*, where these 4 neuron has a strong bias to help to decide the algorithm kind. So we force the network to behave in a certain way.

#figure(
  neural-net(
    layers: (16, 64, 4, 256, 8),
    show-weights: true,
    weight-color: black,
    neuron-radius: 6pt,
    layer-spacing: 50pt,
    label-layers: true,
    max-display-neurons: 8,
    always-show-neuron-count: true
  ),
  caption: [Neural network architecture for algorithm selection with forced decision layer.],
) <CNN>

And the rest of the network make an educated guess and decide the algorithm with the help of *Softmax* among 8 different algorithm mentioned in @tab_algo.

Our NN architecture is trained on a dataset of 50,000 simulated edge computing scenarios, generated by combining and simplyfying the scenarios described in @Bappy2023GoogleCluster and @GWA_T_12_BitBrains. The loss function combines classification accuracy with a custom reward function based on task completion time and energy consumption:

$
L = alpha dot L_"CE" + beta dot L_"perf"
$

where $L_"CE"$ is the cross-entropy loss, $L_"perf"$ is the performance-based loss, and $alpha = 0.6$, $beta = 0.4$ are weighting factors determined through grid search.

#figure(
  table(
    columns: 3,
    [*Input Features*], [*Description*], [*Units*],
    [Base Server/Child Server], [If a server is child or the first server], [Binary 0/1],
    [CPU cycle speed], [Processing speed of the server CPU], [GHz],
    [Current CPU utilization], [Percentage of CPU currently in use], [%],
    [Available memory], [Free RAM available for task processing], [GB],
    [Network latency], [Round-trip time to base server], [ms],
    [Bandwidth capacity], [Maximum data transfer rate], [Mbps],
    [Current queue length], [Number of tasks waiting in queue], [Integer],
    [Average task completion time], [Historical average for completed tasks], [seconds],
    [Power consumption], [Current energy usage of the server], [Watts],
    [Temperature], [CPU temperature indicator], [°C],
    [Storage I/O speed], [Disk read/write performance], [MB/s],
    [Number of active connections], [Current concurrent task connections], [Integer],
    [Server uptime], [Time since last restart], [hours],
    [Geographic distance], [Physical distance from base server], [km],
    [Priority level], [Server tier classification], [1-5],
    [Task type compatibility], [Specialization score for current task], [0-1],
  ),
  caption: "Input features for neural network model"
)
<tab>

#v(4em)

#figure(
  table(
    columns: 2,
    [*Algorithm Name*], [*Type*],
    [Round Robin], [Static],
    [Weighted Round Robin], [Static],
    [IP Hash], [Static],
    [Least Connection], [Dynamic],
    [Weighted Least Connection], [Dynamic],
    [Weighted Response Time], [Dynamic],
    [Resource-Based], [Dynamic],
    [Adaptive Load Balancing], [Hybrid],
  ),
  caption: "Load balancing algorithms in selection pool"
)
<tab_algo>

#v(8em)

== Task Distribution Mechanism

When a task $T_j$ arrives at the base server, the system performs the following steps:

1. *Feature Extraction*: Collect current metrics from all edge servers to form state vector $S_t$

2. *Algorithm Selection*: Feed $S_t$ into the neural network to obtain probability distribution $P = {p_1, p_2, ..., p_8}$ over algorithms

3. *Target Selection*: Selected algorithm determines target edge server $E_"target"$

4. *Task Assignment*: Task $T_j$ is dispatched to $E_"target"$ with metadata including recommended algorithm for potential further distribution

5. *Feedback Collection*: Performance metrics are logged for continuous model improvement

#v(2em)

= Problem Formulation

== Optimization Objective

The primary objective is to minimize the weighted sum of average task completion time and total energy consumption across all edge servers:

$
"minimize" quad J = omega_1 dot T_"avg" + omega_2 dot E_"total"
$

where $T_"avg"$ is the average task completion time, $E_"total"$ is the total energy consumed, and $omega_1$, $omega_2$ are weight coefficients with $omega_1 + omega_2 = 1$.

== Constraints

The optimization is subject to the following constraints:

*Load Balance Constraint:*
$
L_i <= theta dot C_i, quad forall i in {0, 1, ..., N-1}
$

where $theta$ is the load threshold factor (typically 0.85) to prevent server overload.

*Queue Length Constraint:*
$
Q_i <= Q_"max", quad forall i
$

where $Q_"max"$ is the maximum allowable queue length to ensure bounded waiting time.

*Energy Budget Constraint:*
$
sum_(i=0)^(N-1) P_i dot t <= E_"budget"
$

where $P_i$ is the power consumption of server $i$, $t$ is the time period, and $E_"budget"$ is the total energy budget.

*Latency Constraint:*
$
T_"completion"^j <= T_"deadline"^j, quad forall j
$

where $T_"completion"^j$ is the actual completion time and $T_"deadline"^j$ is the deadline for task $j$.

// #v(8em)

// == Mathematical Model

// Let $x_(i j)$ be a binary variable indicating whether task $j$ is assigned to server $i$. Let $a_(i k)$ be a binary variable indicating whether algorithm $k$ is selected for server $i$. The formal optimization problem is:

// $
// min_(x,a) quad &sum_(j=1)^M sum_(i=0)^(N-1) x_(i j) dot (T_"proc"^(i j) + T_"wait"^(i j) + T_"comm"^(i j)) + \ phi &sum_(i=0)^(N-1) P_i dot t_i
// $

// subject to:

// $
// sum_(i=0)^(N-1) x_(i j) &= 1, quad forall j \
// sum_(k=1)^8 a_(i k) &<= 1, quad forall i \
// sum_(j: x_(i j)=1) s_j &<= C_i, quad forall i \
// x_(i j), a_(i k) &in {0, 1}
// $

// where $T_"proc"^(i j)$ is processing time, $T_"wait"^(i j)$ is waiting time, $T_"comm"^(i j)$ is communication time, $s_j$ is the size of task $j$, and $phi$ is the energy cost factor.

#v(2em)

= Methodology

== Simulation Environment

We implemented our system using Python 3.10 with TensorFlow 2.8 for neural network implementation and CloudSim 4.0 for edge computing simulation. The simulation environment consists of 20 heterogeneous edge servers with varying computational capacities ranging from 2.4 GHz to 4.8 GHz, memory from 8 GB to 64 GB, and network latencies from 5 ms to 120 ms.

== Training Procedure

The neural network was trained over 500 epochs with a batch size of 128 using the Adam optimizer with learning rate $eta = 0.001$. We employed a learning rate decay schedule where the rate is multiplied by 0.95 every 50 epochs. The dataset was split into 70% training, 15% validation, and 15% testing sets.

Data augmentation was performed by introducing Gaussian noise ($sigma = 0.05$) to input features and randomly dropping 10% of features during training to improve robustness. Early stopping was implemented with patience of 30 epochs based on validation loss.

== Baseline Comparisons

These baselines are widely used in prior work on load balancing @Zhang2018EdgeSurvey. So we also compare our neuro-guided approach against four most used baseline methods:

1. *Static Round Robin*: Tasks distributed sequentially
2. *Random Selection*: Random algorithm and server selection
3. *Greedy Least Load*: Always select server with minimum current load
4. *Q-Learning Based*: Reinforcement learning approach with state-action table

== Evaluation Metrics

Performance is measured using:
- Average task completion time (seconds)
- Energy consumption (kWh)
- Server utilization variance (lower is better for balance)
- Algorithm selection accuracy (%)
- 95th percentile latency


#v(16em)
= Results and Performance Evaluation

== Overall Performance

Our proposed neuro-guided algorithm selection system achieves significant improvements over baseline methods. @table3 summarizes the key performance metrics.

\

#figure(
  table(
    columns: 6,
    [*Method*], [*Avg. Completion Time (s)*], [*Energy (kWh)*], [*Utilization Variance*], [*Algorithm Accuracy (%)*], [*95th %ile Latency (s)*],
    [Round Robin], [4.82], [12.4], [0.34], [N/A], [8.91],
    [Random], [5.91], [14.2], [0.48], [12.5], [11.2],
    [Greedy], [4.15], [13.1], [0.29], [N/A], [7.83],
    [Q-Learning], [3.74], [11.8], [0.22], [78.3], [6.92],
    [*Proposed*], [*3.18*], [*9.07*], [*0.18*], [*92.3*], [*5.47*],
  ),
  caption: "Performance comparison across different methods"
)<table3>

Our method achieves 34% improvement in average completion time and 27% reduction in energy consumption compared to the static Round Robin approach, while maintaining better load balance across servers as indicated by the lowest utilization variance.


== Scalability Analysis

We evaluated the scalability of our proposed system by varying the number of edge servers from 10 to 100 nodes. @table4 shows that the algorithm selection time remains nearly constant at approximately 12-15 milliseconds even as the network scales. This is because the neural network inference time is independent of the number of servers, depending only on the fixed input feature size.

#figure(
  table(
    columns: 4,
    [*Number of Servers*], [*Selection Time (ms)*], [*Throughput (tasks/s)*], [*Memory Usage (MB)*],
    [10], [12.3], [847], [145],
    [20], [13.1], [1621], [158],
    [50], [14.2], [3892], [183],
    [100], [14.8], [7234], [219],
  ),
  caption: "System performance under different network scales"
)<table4>

The results show that our approach maintains consistent performance characteristics across different deployment scales, making it suitable for both small-scale edge deployments and large distributed edge computing infrastructures. The throughput scales linearly with the number of servers, indicating efficient task distribution without bottlenecks at the decision-making layer.