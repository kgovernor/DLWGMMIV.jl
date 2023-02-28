# DLWGMMIV
```@meta
CurrentModule = DLWGMMIV
```

## Setup

DLWGMMIV can be installed from the package manager in Julia's REPL:

```julia
] add https://github.com/kgovernor/DLWGMMIV.jl.git
```

## De Loecker and Warzynski GMM IV (DLWGMMIV) Production Function Estimation

```@docs
dlwGMMIV
```

## Simulating Firm Data

```@docs
DLWGMMIV.sim_data
```
### Optimization Problem
Let $N_{independent}$ and $N$ be equal to `num_indp_inputs` and `num_inputs`, respectively. Let $X_t = \{x_{1,t}, \dots, x_{N,t}\} \in \mathbb{R}_+^N$ and $X_{t,independent} = \{x_{1,t}, \dots, x_{N_{independent},t}\}$. $N_{independent}$ must be greater than zero and less than $N$, $0 < N_{independent} < N$.

The firm chooses $X_{t,dependent} = X_t \setminus X_{t,independent}$ that optimizes profits in each period $t$, $t \in T = \{0,1,2,\dots\}$.


$$\begin{equation}
\max_{X_{t,dependent}} Profit_t(X_t, \omega_t) \quad \forall t \in T
\end{equation}$$

where $\omega_t$ is the technology parameter, and $Profit_t(X_t, \omega_t) = Output_t(X_t, \omega_t) - Cost_t(X_t)$. The technology parameter $\omega_t$ is subject to a Total Factor Productivity (TFP) shock $\xi_t$, where $\xi_t \sim \mathcal{N}(0, \sigma_{\omega})$.

Exogenous variables at period $t=0$ are given by a random process. Specifically, for $x_{i,0} \in X_{0,independent}$, $x_{i,0} \sim \mathcal{N}(\mu_{x_i}, \sigma_{x_i})$ and $\omega_0 = \xi_0$, where $\mu_{x_i}$ and $\sigma_{x_i}$ equal `indp_inputs_lnmean[i]` and `indp_inputs_lnvariance[i]`. For $t>0$,

$$x_{i,t} = \alpha_{x_i} x_{i,t-1}$$

$$\omega_t = \sum_{i=0}^{3}c_i\omega_{t-1}^i + \xi_t$$

where $\alpha_{x_i} =$ $\sigma_{\alpha} \times$`indp_inputs_params[i]`, $\sigma_{\alpha} \sim \mathcal{N}(0, 1)$, and $c_i =$ `omega_params[i]`.

Let $X_{t}^*$ be the optimal solution to (1). Then,  

$\begin{aligned}
Y_t  &=  Output_t(X_t^*, \omega_t) \\
TC_t &= Cost_t(X_t^*) \\
P_t  &= Y - TC \\
S_t  &= log(Y_t) - \omega_t \\
C_{{x_i},t} &= Cost_t(x_{i,t}^*) \\
rent_{{x_i},t} &= C_{{x_i},t} / x_{i,t}^* \\
share\_TC_{{x_i},t} &= C_{{x_i},t} / TC_t \\
share\_Y_{{x_i},t} &= C_{{x_i},t} / Y_t \\
\end{aligned}$

### Production Functions
#### Cobb Douglas
$$Output_t(X_t, \omega_t) = \exp(\omega_t)\prod_{i=1}^{N}x_{i,t}^{\beta_i}$$

where `prod_params` = $\{\beta_1, \beta_2, \dots, \beta_N\}$.
#### TransLog
$$\begin{align*}
Output_t(X_t, \omega_t) &= \exp(\omega_t + \sum_{i=1}^{N}\beta_ilog(x_{i,t}) \\
&+ \sum_{i=1}^{N}\sum_{j=i}^{N}\beta_{i,j}log(x_{i,t})log(x_{j,t}))
\end{align*}$$

where `prod_params` = $\{\beta_1, \beta_2, \dots, \beta_N,$ 
$\beta_{1,2}, \dots, \beta_{1,N}, \dots, \beta_{i,i+1}, \dots, \beta_{i,N}, \dots, \beta_{N-1,N},$ 
$\beta_{1,1}, \beta_{2,2}, \dots, \beta_{N,N}\}$.
### Cost Functions
#### Constant Elasticity
$$Cost_t(X_t) = \sum_{i=1}^{N}x_{i,t}^{\gamma_i+1}$$

where `cost_params[i]` $= \gamma_i$.

## Examples
### Cobb Douglas Example
Let's first simulate data for industry A with 20,000 firms over 10 periods with an additional initial period 0. The production function in industry A is Cobb-Douglas with two inputs capital, $k$, and labour, $l$ with coefficients 0.1, 0.2, respectively.

Let $k$ be the only independent input, $log(k_0) \sim \mathcal{N}(10, 2)$. 

All other parameters accept default values.

```julia
julia> using DLWGMMIV

julia> input_names = ["k","l"]

julia> num_inputs = length(input_names)

julia> prod_params = [0.1, 0.2]

julia> indp_inputs_lnmean = [10]

julia> indp_inputs_lnvariance = [2]

julia> df = DLWGMMIV.sim_data(20000, 10, 
    num_inputs = num_inputs,
    input_names = input_names, 
    prod_params = prod_params, 
    indp_inputs_lnmean = indp_inputs_lnmean, 
    indp_inputs_lnvariance = indp_inputs_lnvariance
    ).df;

Sim Data for 2 inputs, CD

k Parameters:
  k_prod_params = 0.1 | k_cost_params = 0.0
l Parameters:
  l_prod_params = 0.2 | l_cost_params = 0.15

  First order derivative at optimal L is approximately zero: true

  Second order derivative at optimal L check: true

=======================

SUMMARY:
        100.0% of observations passed first order conditions.
        100.0% of observations passed second order conditions.

=======================
```

Use the GMM IV to estimate the productivity parameters in industry A. Let's set `bstart` to [0.08, 0], assume a Cobb Douglas production function for industry A, and use both the Nelder Mead and LBFGS optimization methods. 

```julia
julia> data = [df.time, df.firm, df.Y, df.k, df.l];

julia> bstart = [0.08, 0];

julia> res_nm = dlwGMMIV(data..., bstart = bstart);

julia> res_lbfgs = dlwGMMIV(data..., bstart = bstart, opt = "LBFGS");

julia> println("
Nelder Mead:\n
Converge = $(res_nm.conv_msg)\n
Objective Value = $(res_nm.other_results.crit)\n
betas = $(res_nm.beta_dlw)
\n
LBFGS:\n
Converge = $(res_lbfgs.conv_msg)\n
Objective Value = $(res_lbfgs.other_results.crit)\n
betas = $(res_lbfgs.beta_dlw)
")

Nelder Mead:

Converge = true

Objective Value = 4.965822136980516e-10

betas = (beta_x1 = 0.09539780702143216, beta_x2 = 0.17476298604850368)


LBFGS:

Converge = true

Objective Value = 1.42422490360313e-19

betas = (beta_x1 = 0.09539780702934003, beta_x2 = 0.17476298621274525)
```

### TransLog Example
Now let's simulate data for industry B with 200 firms over 5 periods with an additional initial period 0. The production function in industry B is TransLog with three inputs capital, $k$, labour, $l$, and material, $m$ with coefficients 0.1, 0.25, 0.2, respectively, and for second order term $km$ the coefficient is 0.01.

All other parameters accept default values.

```julia
julia> using DLWGMMIV

julia> input_names = ["k","l","m"]

julia> num_inputs = length(input_names)

julia> prod_params = [0.1, 0.25, 0.2, 0, 0.01]

julia> df = DLWGMMIV.sim_data(200, 5, 
    num_inputs = num_inputs,
    input_names = input_names, 
    prod_params = prod_params,
    prodF = "tl" 
    ).df;

Sim Data for 3 inputs, tl

k Parameters:
  k_prod_params = 0.1 | k_cost_params = 0.0
l Parameters:
  l_prod_params = 0.25 | l_cost_params = 0.15
m Parameters:
  m_prod_params = 0.2 | m_cost_params = 0.0
kl Parameters:
  kl_prod_params = 0.0 |
km Parameters:
  km_prod_params = 0.01 |
lm Parameters:
  lm_prod_params = 0.0 |
k2 Parameters:
  k2_prod_params = 0.0 |
l2 Parameters:
  l2_prod_params = 0.0 |
m2 Parameters:
  m2_prod_params = 0.0 |

  First order derivative at optimal L is approximately zero: true

  Second order derivative at optimal L check: true

=======================

SUMMARY:
        100.0% of observations passed optimization generating the simulated data.
        100.0% of observations passed first order conditions.
        100.0% of observations passed second order conditions.

=======================
```

Use the GMM IV to estimate the productivity parameters in industry B. Let's set `bstart` to [0.08, 0, 0, 0, 0.015, 0, 0, 0, 0], assume a TransLog production function for industry B, and use both the Nelder Mead and LBFGS optimization methods. 

```julia
julia> data = [df.time, df.firm, df.Y, df.k, df.l, df.m];

julia> bstart = [0.08, 0, 0, 0, 0.015, 0, 0, 0, 0];

julia> res_nm = dlwGMMIV(data..., bstart = bstart, prodF = "tl");

julia> res_lbfgs = dlwGMMIV(data..., bstart = bstart, opt = "LBFGS", prodF = "tl");

julia> println("
Nelder Mead:\n
Converge = $(res_nm.conv_msg)\n
Objective Value = $(res_nm.other_results.crit)\n
betas = $(res_nm.beta_dlw)
\n
LBFGS:\n
Converge = $(res_lbfgs.conv_msg)\n
Objective Value = $(res_lbfgs.other_results.crit)\n
betas = $(res_lbfgs.beta_dlw)
")

Nelder Mead:

Converge = false

Objective Value = 79.28554768851731

betas = (beta_x1 = 0.4453756727068648, beta_x2 = 1.2792439223699248, beta_x3 = -0.5006920829608563, beta_x1x2 = -0.21300433972699703, beta_x1x3 = 0.19994522566063533, beta_x2x3 = 0.0881743447457756, beta_x12 = -0.05639552788752075, beta_x22 = 0.08837596941633362, beta_x32 = -0.14734946617896608)


LBFGS:

Converge = false

Objective Value = 0.0015682997593657287

betas = (beta_x1 = 0.5759958463450249, beta_x2 = 13.674180582661819, beta_x3 = -9.992663746360366, beta_x1x2 = 4.593764648836302, beta_x1x3 = -4.5375950132058565, beta_x2x3 = -1.7111532397047498, beta_x12 = 0.11862766042073726, beta_x22 = -7.3662414691120714, beta_x32 = 7.057850742289901)
```
