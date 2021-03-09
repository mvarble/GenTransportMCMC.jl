# Transport

GenTransportMCMC.jl uses an optimized version of the [Trace Transform DSL](https://www.gen.dev/dev/ref/trace_translators/#Trace-Transform-DSL-1) to encode pushforward distributions.
Users may perform the transport map accelerated inference algorithms using either predefined families of maps $\{ \tilde T(\cdot; \gamma) \}_{\gamma \in\mathcal A}$ or those of their own devising, using the *Transport API*.
This API is centered around the following assumptions of the family, which are discussed in the [Mathematical Details](@ref) section.

- There is a fixed integer $d \in \mathbb N$ such that each $\tilde T(\cdot; \gamma): \mathbb R^d \rightarrow \mathbb R^d$.
- Each map $\tilde T(\cdot; \gamma)$ is invertible.
- Each map $\tilde T(\cdot; \gamma)$ has a lower-triangular Jacobian.
- The parameter set $\mathcal A$ for the family satisfies $\mathcal A = \prod_{i=1}^d \mathbb R^{\mathcal J_i}$ for finite sets $\mathcal J_i$ with accompanying collections of functions $(\psi_j)_{j\in\mathcal J_i}$.
- For each $\gamma = (\gamma_1, \ldots, \gamma_d) \in \mathcal A$, the $i$-th component $\tilde T_i(\cdot; \gamma_i)$ of the map $\tilde T(\cdot; \gamma)$ is determined as such.
$$T_i(\theta; \gamma_i) = \sum_{j\in\mathcal J_i} \gamma_{i,j} \psi_j(\theta)$$


## Transport API

```@docs
OrderedBasis
TransportFamily
TransportMap
inverse
sample_cost
```

## Transport Families

```@docs
hermite
HermiteBasis
```
