# Playground guide

Hyperplane has 18 playgrounds and 111 scenarios. Every scenario below links straight to it on [hyperplane.jatinm.com](https://hyperplane.jatinm.com/).

Every playground works the same way. Pick a scenario and it runs itself. Pause, step, scrub or change the speed with the controls (or space and the arrow keys), change any parameter to rerun it, and open **How it works** for the short version of the math.

- **Regression:** [Linear regression](#linear-regression), [Overfitting & regularization](#overfitting--regularization), [Gaussian processes](#gaussian-processes)
- **Optimization:** [Gradient descent](#gradient-descent)
- **Classification:** [Perceptron](#perceptron), [Logistic regression](#logistic-regression), [k-nearest neighbors](#k-nearest-neighbors), [Naive Bayes](#naive-bayes), [Support vector machines](#support-vector-machines), [Decision trees & forests](#decision-trees--forests), [Neural networks](#neural-networks)
- **Clustering:** [k-means](#k-means), [Gaussian mixtures (EM)](#gaussian-mixtures-em), [DBSCAN](#dbscan)
- **Dimensionality reduction:** [PCA](#pca), [t-SNE](#t-sne)
- **Reinforcement learning:** [Multi-armed bandits](#multi-armed-bandits), [Q-learning](#q-learning)

## Regression

### Linear regression

[Open →](https://hyperplane.jatinm.com/play/linear-regression) · 6 scenarios

Least squares made literal: every residual is drawn as a square, and gradient descent shrinks their total area. Also walks the loss bowl in (m, b) space and fits a plane in 3D.

| Scenario | What happens |
| --- | --- |
| [Fit by eye](https://hyperplane.jatinm.com/play/linear-regression?s=by-eye) | Drag the line to shrink the squares by hand, then let gradient descent finish the job. |
| [One outlier](https://hyperplane.jatinm.com/play/linear-regression?s=outlier) | One far-off point's giant square drags the whole fit toward it. |
| [Fit a plane (3D)](https://hyperplane.jatinm.com/play/linear-regression?s=plane) | Two inputs, so the model is a plane. Gradient descent tilts it into place in 3D. |
| [Walk the loss bowl](https://hyperplane.jatinm.com/play/linear-regression?s=landscape) | Each line is a point on the loss surface. Start gradient descent anywhere on it. |
| [Step too big](https://hyperplane.jatinm.com/play/linear-regression?s=too-big) | A learning rate just past the stability limit makes the loss explode. |
| [Center the inputs](https://hyperplane.jatinm.com/play/linear-regression?s=centered) | Centering x turns the long valley into a round bowl, so descent converges fast. |

### Overfitting & regularization

[Open →](https://hyperplane.jatinm.com/play/overfitting) · 7 scenarios

Polynomials of growing degree against held-out test points, early stopping during training, and how ridge and lasso penalties tame the wiggle.

| Scenario | What happens |
| --- | --- |
| [Wiggle to fit](https://hyperplane.jatinm.com/play/overfitting?s=wiggle) | A degree-10 polynomial threads 14 noisy points and whips around between them. |
| [Find the sweet spot](https://hyperplane.jatinm.com/play/overfitting?s=sweet-spot) | Training error keeps falling while test error traces a U. Find its bottom. |
| [Early stopping](https://hyperplane.jatinm.com/play/overfitting?s=early-stopping) | The trend is learned first and the noise much later, so stop when test error bottoms out. |
| [Ridge tames it](https://hyperplane.jatinm.com/play/overfitting?s=ridge) | An L2 penalty calms a degree-15 fit until it becomes too stiff. |
| [Lasso zeros out](https://hyperplane.jatinm.com/play/overfitting?s=lasso) | An L1 penalty sets many coefficients to exactly zero. |
| [More data](https://hyperplane.jatinm.com/play/overfitting?s=more-data) | With 150 points even degree 15 barely wiggles. |
| [Why L1 gives zeros](https://hyperplane.jatinm.com/play/overfitting?s=geometry) | Why L1 gives zeros: the loss contours meet the diamond at a corner. |

### Gaussian processes

[Open →](https://hyperplane.jatinm.com/play/gaussian-processes) · 7 scenarios

Regression that knows where it is unsure: a posterior mean with an uncertainty band that pinches at each observation. Ends with Bayesian optimization running on its own.

| Scenario | What happens |
| --- | --- |
| [One point at a time](https://hyperplane.jatinm.com/play/gaussian-processes?s=stream) | Observations arrive one at a time, and the band pinches and the samples snap to each. |
| [Add observations](https://hyperplane.jatinm.com/play/gaussian-processes?s=observe) | Click to add data. Uncertainty vanishes at observations and returns far from them. |
| [Length scale](https://hyperplane.jatinm.com/play/gaussian-processes?s=length-scale) | A small ℓ means nearby points barely relate, so the samples go wild. |
| [Noisy measurements](https://hyperplane.jatinm.com/play/gaussian-processes?s=noise) | With noise, the mean stops interpolating and the uncertainty never fully vanishes. |
| [Kernels encode beliefs](https://hyperplane.jatinm.com/play/gaussian-processes?s=periodic) | A periodic kernel extrapolates confidently because it assumes the pattern repeats. |
| [Let the data choose](https://hyperplane.jatinm.com/play/gaussian-processes?s=fit) | Maximize the marginal likelihood to let the data choose ℓ and σ². |
| [Bayesian optimization](https://hyperplane.jatinm.com/play/gaussian-processes?s=bayes-opt) | Bayesian optimization finds the max of a hidden function, sampling where μ + κσ peaks. |

## Optimization

### Gradient descent

[Open →](https://hyperplane.jatinm.com/play/gradient-descent) · 9 scenarios

Gradient descent, momentum, RMSProp and Adam race across 2D and orbitable 3D loss surfaces, plus 1D views of step size and local minima.

| Scenario | What happens |
| --- | --- |
| [Narrow valley](https://hyperplane.jatinm.com/play/gradient-descent?s=valley) | Plain GD zig-zags across a narrow valley while momentum and Adam cut through. |
| [Banana valley](https://hyperplane.jatinm.com/play/gradient-descent?s=rosenbrock) | The banana valley is easy to fall into and slow to follow. |
| [Four minima](https://hyperplane.jatinm.com/play/gradient-descent?s=four-minima) | Which basin each optimizer lands in depends on where it starts. |
| [Escape the saddle](https://hyperplane.jatinm.com/play/gradient-descent?s=saddle) | Starting on a saddle's ridge: who escapes first? |
| [Bumpy landscape](https://hyperplane.jatinm.com/play/gradient-descent?s=egg-crate) | Many local minima. Can momentum carry an optimizer over the bumps? |
| [In 3D](https://hyperplane.jatinm.com/play/gradient-descent?s=3d) | The same race on an orbitable 3D surface. |
| [Bumpy, in 3D](https://hyperplane.jatinm.com/play/gradient-descent?s=3d-bumpy) | Hills and dips in 3D, where momentum can jump a ridge into a deeper basin. |
| [Overshoot (1D)](https://hyperplane.jatinm.com/play/gradient-descent?s=overshoot) | 1D: past η = 0.5 the steps oscillate, and past η = 1 they diverge. |
| [Stuck in a dip (1D)](https://hyperplane.jatinm.com/play/gradient-descent?s=local-min) | 1D: gradient descent settles into the shallower well. |

## Classification

### Perceptron

[Open →](https://hyperplane.jatinm.com/play/perceptron) · 5 scenarios

A line that fixes itself one mistake at a time, shown side by side in data space and weight space, with the convergence guarantee as a meter.

| Scenario | What happens |
| --- | --- |
| [Separable data](https://hyperplane.jatinm.com/play/perceptron?s=separable) | The boundary jumps after each mistake, then locks in. |
| [Razor-thin gap](https://hyperplane.jatinm.com/play/perceptron?s=thin-margin) | A smaller gap means more mistakes. Compare the count against the guarantee. |
| [Noisy labels](https://hyperplane.jatinm.com/play/perceptron?s=noisy) | With flipped labels no line separates the data, so the perceptron thrashes forever. |
| [Weight space](https://hyperplane.jatinm.com/play/perceptron?s=weight-space) | Each update is a vector addition that walks 𝐰 into the feasible region. |
| [Classify by hand](https://hyperplane.jatinm.com/play/perceptron?s=by-hand) | Drag the weight vector yourself until there are zero mistakes. |

### Logistic regression

[Open →](https://hyperplane.jatinm.com/play/logistic-regression) · 5 scenarios

Turning a hard boundary into calibrated probabilities: the sigmoid in 1D, 2D and 3D, per-point cross-entropy, and why separable data needs L2.

| Scenario | What happens |
| --- | --- |
| [Overlapping classes](https://hyperplane.jatinm.com/play/logistic-regression?s=overlap) | The boundary settles where the classes meet, with probability bands on either side. |
| [Perfectly separable](https://hyperplane.jatinm.com/play/logistic-regression?s=separable) | On separable data ‖w‖ grows forever and the sigmoid becomes a hard step. |
| [Rein it in with L2](https://hyperplane.jatinm.com/play/logistic-regression?s=l2) | An L2 penalty levels ‖w‖ off and keeps the probabilities soft. |
| [The sigmoid in 3D](https://hyperplane.jatinm.com/play/logistic-regression?s=3d) | The probability as a 3D sheet that tilts into place, then steepens. |
| [One dimension](https://hyperplane.jatinm.com/play/logistic-regression?s=sigmoid) | Shape the 1D sigmoid by hand and watch each point's penalty. |

### k-nearest neighbors

[Open →](https://hyperplane.jatinm.com/play/k-nearest-neighbors) · 5 scenarios

Each cell of the plane is classified by its neighbors' vote, with k trading noise for blur and distance metrics changing what 'near' means.

| Scenario | What happens |
| --- | --- |
| [k = 1 memorizes](https://hyperplane.jatinm.com/play/k-nearest-neighbors?s=k1) | k = 1 memorizes every point, noise included: 100% train accuracy, islands everywhere. |
| [Smooth it out](https://hyperplane.jatinm.com/play/k-nearest-neighbors?s=smooth) | At k = 15 the noise is outvoted. Hover to see who votes. |
| [Too smooth](https://hyperplane.jatinm.com/play/k-nearest-neighbors?s=too-smooth) | On a spiral, a large k blurs the arms away. |
| [Shape of “near”](https://hyperplane.jatinm.com/play/k-nearest-neighbors?s=metric) | Euclidean, Manhattan or Chebyshev: a circle, a diamond or a square. |
| [Draw your own](https://hyperplane.jatinm.com/play/k-nearest-neighbors?s=draw) | Click to plant your own points of each class. |

### Naive Bayes

[Open →](https://hyperplane.jatinm.com/play/naive-bayes) · 5 scenarios

One Gaussian per class learned from a stream of points, curved decision boundaries, what the naive assumption gives up compared with QDA, and the class densities as 3D hills.

| Scenario | What happens |
| --- | --- |
| [Learn from a stream](https://hyperplane.jatinm.com/play/naive-bayes?s=stream) | Training is just means and spreads, so watch the ellipses settle point by point. |
| [Curved boundaries](https://hyperplane.jatinm.com/play/naive-bayes?s=curved) | Different spreads per class give a curved, quadratic boundary. |
| [The naive assumption](https://hyperplane.jatinm.com/play/naive-bayes?s=naive) | Correlated features break the axis-aligned assumption. Toggle full covariance (QDA). |
| [Three classes](https://hyperplane.jatinm.com/play/naive-bayes?s=three) | Three classes, each winning wherever it is most probable. |
| [Densities in 3D](https://hyperplane.jatinm.com/play/naive-bayes?s=hills) | Each class is a hill of probability, and the boundary is where the hills cross. |

### Support vector machines

[Open →](https://hyperplane.jatinm.com/play/svm) · 7 scenarios

SMO trains the SVM pair by pair until the widest street appears. Then soft margins, outliers, RBF and polynomial kernels, and the kernel trick lifted into 3D.

| Scenario | What happens |
| --- | --- |
| [Widest street](https://hyperplane.jatinm.com/play/svm?s=widest-street) | Only the support vectors hold the street in place. Drag the others and nothing moves. |
| [Soft margin](https://hyperplane.jatinm.com/play/svm?s=soft-margin) | Overlapping classes: C trades a wide, tolerant street for fewer violations. |
| [One outlier](https://hyperplane.jatinm.com/play/svm?s=outlier) | A huge C contorts the boundary around one outlier. |
| [RBF on rings](https://hyperplane.jatinm.com/play/svm?s=rbf-rings) | An RBF kernel curves the boundary around the inner ring. |
| [γ too high](https://hyperplane.jatinm.com/play/svm?s=gamma) | A huge γ wraps individual points, which is overfitting. |
| [Polynomial XOR](https://hyperplane.jatinm.com/play/svm?s=poly-xor) | A degree-2 kernel adds x·y, which is exactly what XOR needs. |
| [Kernel trick in 3D](https://hyperplane.jatinm.com/play/svm?s=kernel-trick) | Lift points to x² + y², where a flat plane separates the rings. |

### Decision trees & forests

[Open →](https://hyperplane.jatinm.com/play/decision-trees) · 6 scenarios

A tree grows split by split while its diagram builds alongside. Shows depth and overfitting, axis-aligned staircases, and a random forest voting.

| Scenario | What happens |
| --- | --- |
| [Grow a tree](https://hyperplane.jatinm.com/play/decision-trees?s=grow) | Greedy splits, one yes/no question at a time. Hover to trace a point down the tree. |
| [Too deep](https://hyperplane.jatinm.com/play/decision-trees?s=too-deep) | Depth 14 boxes in every noisy point. Check the test curve. |
| [Diagonal boundary](https://hyperplane.jatinm.com/play/decision-trees?s=diagonal) | Axis-aligned cuts approximate a diagonal with a staircase. |
| [Greedy blindness](https://hyperplane.jatinm.com/play/decision-trees?s=xor) | No single cut helps on XOR, so greedy splitting can't see two moves ahead. |
| [Forest vs. tree](https://hyperplane.jatinm.com/play/decision-trees?s=forest) | Many randomized trees voting beat one jagged deep tree. |
| [Forest on a spiral](https://hyperplane.jatinm.com/play/decision-trees?s=forest-spiral) | Enough stacked boxes can still follow a spiral. |

### Neural networks

[Open →](https://hyperplane.jatinm.com/play/neural-networks) · 9 scenarios

Train a network live with every neuron's response drawn inside the diagram, watch layers bend space, and see the output fold as a 3D surface.

| Scenario | What happens |
| --- | --- |
| [Untangle a spiral](https://hyperplane.jatinm.com/play/neural-networks?s=spiral) | Each little square is one neuron's response, and the early edges combine into a spiral. |
| [Fence in a circle](https://hyperplane.jatinm.com/play/neural-networks?s=three-neurons) | Three hidden neurons are three lines, exactly enough to fence in a circle. |
| [No nonlinearity](https://hyperplane.jatinm.com/play/neural-networks?s=linear) | With linear activations depth doesn't help, so XOR is impossible. |
| [ReLU folds](https://hyperplane.jatinm.com/play/neural-networks?s=relu) | ReLU builds boundaries from straight folds. |
| [Output surface in 3D](https://hyperplane.jatinm.com/play/neural-networks?s=surface) | The output as a 3D sheet folding into a dome over the inner ring. |
| [Folding a spiral (3D)](https://hyperplane.jatinm.com/play/neural-networks?s=surface-spiral) | A spiral needs ridges that wind around the center. |
| [Layers bend space](https://hyperplane.jatinm.com/play/neural-networks?s=bend) | A 2-2-2-2-1 network morphs the plane layer by layer until a line separates it. |
| [Rings can't be untangled](https://hyperplane.jatinm.com/play/neural-networks?s=rings) | Two neurons per layer can stretch the plane but not tear it. |
| [One neuron](https://hyperplane.jatinm.com/play/neural-networks?s=neuron) | One neuron: a tilted ramp, bent by an activation. Hover to trace a point. |

## Clustering

### k-means

[Open →](https://hyperplane.jatinm.com/play/k-means) · 6 scenarios

Lloyd's algorithm phase by phase (assign, then update), bad initializations, the wrong k, and the elbow method.

| Scenario | What happens |
| --- | --- |
| [Four blobs](https://hyperplane.jatinm.com/play/k-means?s=blobs) | Assign to the nearest centroid, move each centroid to the mean, repeat. |
| [Bad start](https://hyperplane.jatinm.com/play/k-means?s=bad-start) | Huddled starting centroids converge to a poor local minimum. Try k-means++. |
| [Wrong k](https://hyperplane.jatinm.com/play/k-means?s=wrong-k) | Six centroids for four blobs, so blobs get split. |
| [Moons break it](https://hyperplane.jatinm.com/play/k-means?s=moons) | Straight Voronoi boundaries slice across non-round clusters. |
| [No structure at all](https://hyperplane.jatinm.com/play/k-means?s=uniform) | Uniform noise still gets carved into k confident cells. |
| [How many clusters?](https://hyperplane.jatinm.com/play/k-means?s=elbow) | Inertia always falls with k. Find the elbow. |

### Gaussian mixtures (EM)

[Open →](https://hyperplane.jatinm.com/play/gaussian-mixtures) · 6 scenarios

EM fits ellipses that shift, rotate and stretch, assigns soft membership, and is compared with k-means on the same data.

| Scenario | What happens |
| --- | --- |
| [Stretched clusters](https://hyperplane.jatinm.com/play/gaussian-mixtures?s=stretched) | Ellipses shift, rotate and stretch to fit long, thin clusters. |
| [Soft membership](https://hyperplane.jatinm.com/play/gaussian-mixtures?s=soft) | Points in the overlap get blended colors because the model is unsure. |
| [Big & small](https://hyperplane.jatinm.com/play/gaussian-mixtures?s=sizes) | Mixing weights and covariances let components differ in size. |
| [k-means vs. GMM](https://hyperplane.jatinm.com/play/gaussian-mixtures?s=vs-kmeans) | Same data, same k: straight cuts against fitted shapes. |
| [Too many components](https://hyperplane.jatinm.com/play/gaussian-mixtures?s=too-many) | Extra components split clusters, yet the likelihood still improves. |
| [Non-Gaussian shapes](https://hyperplane.jatinm.com/play/gaussian-mixtures?s=moons) | Ellipses can't bend, but more of them tile a moon. |

### DBSCAN

[Open →](https://hyperplane.jatinm.com/play/dbscan) · 6 scenarios

Clusters spread from core point to core point, follow any shape and leave noise out. Shows how ε and minPts break it.

| Scenario | What happens |
| --- | --- |
| [Moons, done right](https://hyperplane.jatinm.com/play/dbscan?s=moons) | Clusters follow any shape, including the moons k-means slices in half. |
| [Rings](https://hyperplane.jatinm.com/play/dbscan?s=rings) | Nested rings separate without being told how many clusters there are. |
| [Noise stays noise](https://hyperplane.jatinm.com/play/dbscan?s=noise) | Sparse background points are labeled noise, not forced into clusters. |
| [ε too small](https://hyperplane.jatinm.com/play/dbscan?s=eps-small) | A tiny ε gives fragments and noise everywhere. |
| [ε too big](https://hyperplane.jatinm.com/play/dbscan?s=eps-big) | A huge ε merges everything. The k-distance elbow suggests a better value. |
| [Mixed density](https://hyperplane.jatinm.com/play/dbscan?s=density) | Mixed densities: no single ε works for both clusters. |

## Dimensionality reduction

### PCA

[Open →](https://hyperplane.jatinm.com/play/pca) · 5 scenarios

Power iteration finds the direction of greatest variance, the data rotates into its principal axes, and a 3D cloud is squashed onto its best plane.

| Scenario | What happens |
| --- | --- |
| [Find the best line](https://hyperplane.jatinm.com/play/pca?s=best-line) | Rotate a line to keep the most variance, then let power iteration find it. |
| [A round cloud](https://hyperplane.jatinm.com/play/pca?s=round) | In a round cloud every direction is equally good. |
| [Almost a line](https://hyperplane.jatinm.com/play/pca?s=needle) | In a needle-thin cloud, one number per point loses almost nothing. |
| [3D → 2D](https://hyperplane.jatinm.com/play/pca?s=3d) | A tilted 3D cloud squashed onto its best plane. |
| [Rotate, then project](https://hyperplane.jatinm.com/play/pca?s=rotate) | Center, rotate onto the principal axes, then drop one. |

### t-SNE

[Open →](https://hyperplane.jatinm.com/play/t-sne) · 6 scenarios

t-SNE optimizes live next to PCA's best linear view on high-dimensional data, with scenarios on perplexity and on the cluster sizes and distances that mislead.

| Scenario | What happens |
| --- | --- |
| [Clusters PCA can't see](https://hyperplane.jatinm.com/play/t-sne?s=hidden) | Six clusters in 10D that PCA piles up and t-SNE separates. |
| [Perplexity](https://hyperplane.jatinm.com/play/t-sne?s=perplexity) | At perplexity 3 clusters shatter. Slide it up to bring the structure back. |
| [Cluster sizes lie](https://hyperplane.jatinm.com/play/t-sne?s=sizes) | A 10× wider cluster is drawn the same size. |
| [Distances lie](https://hyperplane.jatinm.com/play/t-sne?s=distances) | Gaps between islands don't reflect true distances. |
| [Groups of groups](https://hyperplane.jatinm.com/play/t-sne?s=hierarchy) | Groups of groups: islands sit near their siblings. |
| [Swiss roll](https://hyperplane.jatinm.com/play/t-sne?s=swiss) | Unrolling a swiss roll that PCA squashes flat. |

## Reinforcement learning

### Multi-armed bandits

[Open →](https://hyperplane.jatinm.com/play/bandits) · 5 scenarios

Greedy, ε-greedy, UCB and Thompson sampling pull the same slot machines at once, with each strategy's beliefs and cumulative regret.

| Scenario | What happens |
| --- | --- |
| [Three slot machines](https://hyperplane.jatinm.com/play/bandits?s=thompson) | Thompson sampling's beliefs about bad arms freeze as they stop being worth trying. |
| [Greedy gets stuck](https://hyperplane.jatinm.com/play/bandits?s=greedy) | One unlucky first pull can lock greedy onto a bad arm forever. |
| [Optimism (UCB)](https://hyperplane.jatinm.com/play/bandits?s=ucb) | Optimism: neglected arms' confidence bars grow until they get tried. |
| [Close call](https://hyperplane.jatinm.com/play/bandits?s=close) | Payouts of 45 to 55% take hundreds of pulls to tell apart. |
| [Many arms](https://hyperplane.jatinm.com/play/bandits?s=many) | With eight arms, constant ε keeps wasting pulls while the adaptive strategies stop. |

### Q-learning

[Open →](https://hyperplane.jatinm.com/play/q-learning) · 6 scenarios

An agent learns a gridworld you can paint, with value spreading back from the goal and the greedy policy drawn as arrows. Ends with value iteration for when the map is known.

| Scenario | What happens |
| --- | --- |
| [Find the goal](https://hyperplane.jatinm.com/play/q-learning?s=maze) | Value leaks back from the goal until the arrows form a route. |
| [Cliff walk](https://hyperplane.jatinm.com/play/q-learning?s=cliff) | The greedy path hugs the cliff. Add wind and it backs off. |
| [Explore vs. exploit](https://hyperplane.jatinm.com/play/q-learning?s=explore) | With ε = 0 the agent settles for the nearby small reward. |
| [Short-sighted agent](https://hyperplane.jatinm.com/play/q-learning?s=discount) | A low γ makes the far +1 not worth the trip. |
| [Plan with value iteration](https://hyperplane.jatinm.com/play/q-learning?s=value-iteration) | Knowing the map, Bellman sweeps spread value outward. Paint walls to replan. |
| [Build a world](https://hyperplane.jatinm.com/play/q-learning?s=build) | An empty room for painting walls, pits and goals while the agent explores. |
