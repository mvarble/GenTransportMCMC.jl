# Mathematical Details

This library is entirely focused on implementing the *transport map accelerated Markov chain Monte Carlo* inference algorithm discussed in the following paper.


> Parno, Matthew D., and Youssef M., Marzouk. "Transport Map Accelerated Markov Chain Monte Carlo". *SIAM/ASA Journal on Uncertainty Quantification* 6, no.2 (2018): 645-682. [SIAM Link](https://doi.org/10.1137/17M1134640), [arXiv Link](https://arxiv.org/abs/1412.5492).


In summary, this is an inference algorithm which samples a posterior distribution ``\mu_\theta`` by simultaneously finding a map ``T`` which transports said distribution to a *reference* standard Gaussian ``\mu_r = T_\# \mu_\theta`` and performing Metropolis-Hastings in this reference space.
The proceeding subsections will describe the objects at play.

## Transport maps

Suppose we have some distribution ``\mu_\theta`` with density ``\pi`` with respect to the Lebesgue measure on ``\mathbb R^d``.

```math
\displaystyle \mu_\theta({\rm d}\theta) = \pi(\theta) {\rm d}\theta
```

Provided a measurable map ``T: \mathbb R^d \rightarrow \mathbb R^d``, we may *transport* ``\mu_\theta`` to another distribution ``T_\#\mu_\theta``, defined to act on measurable functions ``f: \mathbb R^d \rightarrow \mathbb R`` like so.

```math
\displaystyle \int f(r) T_\#\mu_\theta({\rm d}r) = \int f\big(T(\theta)\big) \mu_\theta({\rm d}\theta)
```

!!! note "Note"
    From a Monte Carlo perspective, a sample ``r \sim T_\#\mu_\theta`` is equivalent to sampling ``\theta \sim \mu_\theta`` and taking ``r = T(\theta)`` (hence the phrase *Transport*).

If our density ``\pi`` is continuous and ``T`` is a continuously differentiable bijection, the change of variables theorem from calculus tells us that the transport distribution (let's denote this ``\mu_r = T_\#\mu_\theta``) will have the following density ``p``.

```math
p(r) = \pi\big(T^{-1}(r)\big) \big|\det \nabla T^{-1}(r)\big| \\[1em]
\displaystyle\int f(r) \mu_r({\rm d}r) = \int f(r) \pi\big(T^{-1}(r)\big) \big| \det \nabla T^{-1}(r) \big| {\rm d}r
```

Above and throughout, ``\nabla G`` denotes the Jacobian matrix of a map ``G: \mathbb R^d \rightarrow \mathbb R^d``.
From the perspective of the Gen ecosystem, if an address `:theta` is intended to encode samples of ``\mu_\theta``, then one can subsequently encode samples of ``\mu_r = T_\ast\mu_\theta`` to an address `:r` with the [Trace Transform DSL](https://www.gen.dev/dev/ref/trace_translators/#Trace-Transform-DSL-1).

The paper above concerns itself with finding a transport map (referred to as the *Knothe-Rosenblatt rearrangement*) ``T: \mathbb R^d \rightarrow \mathbb R^d`` such that ``\nabla T`` is lower-triangular and ``\mu_r = T_\#\mu_\theta`` is the standard Gaussian measure on ``\mathbb R^d``.
The lower-triangular property of ``\nabla T`` is equivalent to ``T`` having the following structure.

```math
T(\theta_1, \ldots, \theta_d) = \Big( T_1(\theta_1), T_2(\theta_1, \theta_2), \ldots, T_d(\theta_1, \ldots, \theta_d) \Big)
```

Such a structure is computationally advantageous, as the inverse image ``T^{-1}(r)`` and Jacobian determinant ``\det \nabla T(\theta)`` are easy to evaluate.
Also, the Gaussian nature of ``\mu_r = T_\#\mu_\theta`` means that sampling ``\theta \sim \mu_\theta`` is as easy as sampling from standard Gaussian ``r \sim \mu_r`` and evaluating ``\theta = T^{-1}(r)``.
Hence, having such a map ``T``, or a nice approximation thereof, means we may efficiently sample complicated posterior distributions ``\mu_\theta``; such an algorithm is apt for systems like Gen.

## Approximating transport maps

In practice, it is infeasible to actually get the transport constraint ``T_\#\mu_\theta = \mu_r`` for a standard Gaussian ``\mu_r``.
Thus, for a fixed distribution ``\mu_\theta`` and proposed transport map ``\tilde T``, it is imperative to measure the discrepancy of our transport from the standard Gaussian.
In other words, we want to measure the effectiveness of the following approximation.

```math
\tilde T_\#\mu_\theta \approx \mu_r
```

One solution to this is to recognize that the true constraint ``\tilde T_\#\mu_\theta = \mu_r`` is equivalent to ``\tilde T^{-1}_\#\mu_r = \mu_\theta``; this way, we may consider the discrepancy of an equivalent approximation.

```math
\mu_\theta \approx \tilde T_\#^{-1}\mu_r
```

Denoting ``\tilde\pi`` as the density of ``\tilde T_\#^{-1}\mu_r``, we have

```math
\tilde\pi(\theta) = p\big(\tilde T(\theta)\big) \big|\det\nabla\tilde T(\theta)\big|.
```

From here, we may measure the discrepancy between ``\mu_\theta`` and ``\tilde T_\#^{-1}\mu_r`` with the Kullback-Leibler divergence.

```math
\begin{aligned}
  D_{KL}\big( \mu_\theta \parallel \tilde T_\#^{-1}\mu_r \big)
  &= \int \log\Big(\frac{\tilde\pi(\theta)}{\pi(\theta)}\Big) \mu_\theta({\rm d}\theta) \\
  &= \int \log\pi(\theta) \mu_\theta({\rm d}\theta) + \int \Big( - \log p\big(\tilde T(\theta)\big) - \log\big|\det\nabla\tilde T(\theta)\big| \Big) \mu_\theta({\rm d}\theta)
\end{aligned}
```

This divergence is minimized when our transport constraint is exact, and so finding the true transport ``T`` is equivalent to solving the following optimization problem over the set ``\mathcal T`` of lower-triangular continuously differentiable bijections.

```math
T = \argmin_{\tilde T \in \mathcal T} \int \Big( - \log p\big(\tilde T(\theta)\big) - \log\big|\det\nabla\tilde T(\theta)\big| \Big) \mu_\theta({\rm d}\theta)
```

If provided samples ``\theta^{(1)}, \ldots, \theta^{(K)}`` from ``\mu_\theta``, we may approximate the integral above as follows.

```math
\begin{aligned}
  &\int \Big( - \log p\big(\tilde T(\theta)\big) - \log\big|\det\nabla\tilde T(\theta)\big| \Big) \mu_\theta({\rm d}\theta) \\
  &\quad\approx K^{-1}\sum_{k=1}^K \Big( -\log p\big(\tilde T(\theta^{(k)})\big) - \log\big|\det\nabla\tilde T(\theta^{(k)})\big| \Big) \\
  &\quad= K^{-1}\sum_{k=1}^K \Big( \frac{n}{2}\log(2\pi) + \frac{1}{2}\sum_{i=1}^d \tilde T_i^2(\theta) - \sum_{i=1}^d \log\frac{\partial \tilde T_i}{\partial \theta_i}(\theta^{(k)}) \Big) \\
  &\quad= \frac{n}{2}\log(2\pi) + K^{-1} \sum_{i=1}^d \sum_{k=1}^K \Big( \frac{1}{2}\tilde T_i^2(\theta) - \log\frac{\partial\tilde T_i}{\partial \theta_i}(\theta^{(k)}) \Big)
\end{aligned}
```

Note that the first equality above is utilizing the closed form of the standard Gaussian density ``p`` and the lower-triangular structure of ``\nabla\tilde T(\theta)``.
With this approximation, we choose to instead minimize the following objective function.

```math
C(\tilde T) = \sum_{i=1}^d\sum_{k=1}^K \Big(\frac{1}{2} \tilde T_i^2(\theta^{(k)}) - \log\frac{\partial\tilde T_i}{\partial\theta_i}(\theta^{(k)}) \Big)
```

From here, we may reduce the optimization problem from the large space ``\mathcal T`` to a parameterized set of maps ``\{T(\cdot; \gamma)\}_{\gamma \in \mathcal A}`` where ``\mathcal A`` is some nice parameter set.
In particular, for each ``i=1,\ldots, d``, we may pick a finite ordered basis of functions ``(\psi_j)_{j\in\mathcal J_i}`` and declare our map parameterization so that our parameter set is ``\mathcal A = \prod_{i=1}^d \mathbb R^{\mathcal J_i}`` and the components of the map ``T(\cdot; \gamma)`` are as follows.

```math
\tilde T_i(\theta; \gamma_i) = \sum_{j\in\mathcal J_i} \gamma_{i,j} \psi_j(\theta)
```

This linear form makes optimizing ``C\big(\tilde T(\cdot;\gamma)\big)`` simpler.

## Map-based Markov chain Monte Carlo

Provided a complicated posterior distribution ``\mu_\theta`` and a transport map ``\tilde T`` such that ``\tilde T_\#\mu_\theta`` is approximately Guassian, we may choose to perform the Metropolis-Hastings algorithm on either of ``\mu_\theta`` or ``\tilde T_\#\mu_\theta`` and subsequently map samples via ``\tilde T``.
Because ``\tilde T_\#\mu_\theta`` is approximately Gaussian, our proposals do not need to account for complicated features of the distribution, as they would for ``\mu_\theta``. 
To this end, the map ``\tilde T`` is accounting for these complicated features.

For a fixed proposal kernel ``Q_r`` with density ``q_r``

```math
Q_r({\rm d}r' | r) = q_r(r' | r) {\rm d}r',
```

the induced ``\tilde T``-pullback kernel ``Q_\theta=\tilde T^{-1}_\#Q_r``, characterized to satisfy the following for all measurable ``f``

```math
\displaystyle\int f(\theta') Q_\theta\big({\rm d}\theta' | \theta \big) = \int f(r') Q_r\big({\rm d}r' | \tilde T(\theta) \big),
```

can be evaluated as follows.

```math
Q_\theta\big({\rm d}\theta' | \theta \big) = q_r\big( \tilde T(\theta') | \tilde T(\theta) \big) \big|\det\nabla\tilde T(\theta')\big| {\rm d}\theta'
```

Performing Metropolis-Hastings for ``\mu_\theta`` associated to proposal ``Q_\theta`` now amounts to the following algorithm.

1. Get reference ``r = \tilde T(\theta)``
2. Sample proposal ``r' \sim Q_r(\cdot | r)``
3. Solve target ``\theta' = \tilde T^{-1}(r')``
4. Perform accept-reject step with acceptance probability ``\alpha``
```math
\alpha = \min\Big\{ 1, \frac{\pi(\theta')q_r\big(\tilde T(\theta) | \tilde T(\theta') \big)\big|\det\nabla\tilde T(\theta)\big|}{\pi(\theta)q_r\big(\tilde T(\theta') | \tilde T(\theta) \big)\big|\det\nabla\tilde T(\theta')\big|} \Big\}
```

If ``Q_r`` is a derivative-based proposal, we can get the gradient of the density ``\tilde p`` for distribution ``\tilde T_\#\mu_\theta`` by recognizing

```math
\log\tilde p(r) = \log\pi\big(\tilde T ^{-1}(r)\big) + \sum_{i=1}^n \log\frac{\partial\tilde T_i^{-1}}{\partial r_i}(r)
```

and so the chain-rule gives us

```math
\nabla_\theta \log\tilde p\big( \tilde T(\theta) \big) = \nabla \log\pi(\theta) - \sum_{i=1}^n \frac{e_i^T H\tilde T(\theta)}{\frac{\partial \tilde T_i}{\partial \theta_i}(\theta)}
```

where ``H\tilde T(\theta)`` is the Hessian matrix associated to ``\tilde T`` at ``\theta``.

## Adaptive transport map Markov chain Monte Carlo

The algorithm now amounts to performing Metropolis-Hastings with a transport map ``\tilde T(\cdot; \gamma)`` to produce samples of ``\mu_\theta`` and adaptively updating the transport map ``\tilde T(\cdot; \gamma) \leadsto \tilde T(\cdot; \gamma')`` from said samples.
Below is pseudocode for implementing the algorithm.

```@raw html
<div class="admonition is-info">
  <header class="admonition-header">Transport map accelerated Monte Carlo</header>
  <div style="display: flex; flex-direction: column; margin: 0 1em 0.5em 1em">
    <div style="display: flex; margin: 0.5em 0 0.5em 0.5em">
      <em style="margin-right: 1em">Input.</em>
      <span>Initial state $\theta_0$, inital vector of transport map parameters $\overline \gamma_0 \in \mathcal A$, reference proposal $Q_r$, number of steps $K_U$ between map adaptations, total number of steps $L$.<span>
    </div>
    <div style="display: flex; margin: 0.5em 0 0.5em 0.5em">
      <em style="margin-right: 1em">Output.</em>
      <span>MCMC samples $\{\theta^{(1)}, \ldots, \theta^{(L)}\}$ of the target distribution $\mu_\theta$.</span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>1</strong></tt>
      <span>Set state $\theta^{(1)} = \theta_0$</span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>2</strong></tt>
      <span>Set parameters $\overline\gamma^{(1)} = \overline\gamma_0$</span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>3</strong></tt>
      <span><strong style="margin-right: 0.5em">for</strong>$k \leftarrow 1 \ldots L-1$<strong style="margin-left: 0.5em">do</strong></span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>4</strong></tt>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span>Compute the reference state, $r^{(k)} = \tilde T(\theta^{(k)}; \overline \gamma^{(k)})$</span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>5</strong></tt>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span>Sample the reference proposal $r' \sim Q_r(\cdot|r^{(k)})$</span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>6</strong></tt>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span>Compute the target proposal sample, $\theta' = \tilde T^{-1}(r'; \overline \gamma^{(k)})$</span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>7</strong></tt>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span>
        Calculate the acceptance probability:
        \[ \alpha = \min\Big\{  1, \frac{\pi(\theta')q_r\big(r^{(k)} | r' \big)\big|\det\nabla\tilde T(\theta^{(k)})\big|}{\pi(\theta)q_r\big(r' | \tilde r^{(k)} \big)\big|\det\nabla\tilde T(\theta')\big|}\Big\} \]
      </span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>8</strong></tt>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span>Set $\theta^{(k+1)}$ to $\theta'$ with probability $\alpha$; else set $\theta^{(k+1)}$ to $\theta^{(k)}$</span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>9</strong></tt>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span><strong style="margin-right: 0.5em">if</strong>$(k \mod K_U) = 0$<strong style="margin-left: 0.5em">then</strong></span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>10</strong></tt>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span><strong style="margin-right: 0.5em">for</strong>$i \leftarrow 1\ldots d$<strong style="margin-left: 0.5em">do</strong></span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>11</strong></tt>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span>Update $\overline \gamma_i^{(k+1)}$ by optimizing $\gamma \rightarrow \tilde C\big(\tilde T(\cdot; \gamma)\big)$ for samples $\{\theta^{(1)},\ldots,\theta^{(k)}\}$</span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>12</strong></tt>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span><strong style="margin-right: 0.5em">else</strong></span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>13</strong></tt>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span style="margin: 0 1em 0 0.5em; border-left: 1px solid black"></span>
      <span>Keep $\overline \gamma^{(k+1)} = \overline \gamma^{(k)}$</span>
    </div>
    <div style="display: flex">
      <tt style="width: 1.5em; text-align: right; margin-right: 1em;"><strong>14</strong></tt>
      <span><strong style="margin-right:1em">return</strong> $\{\theta^{(1)},\ldots,\theta^{(L)}\}$</span>
    </div>
  </div>
</div>
```
