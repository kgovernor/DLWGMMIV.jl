###########################
# Generate Simulated Data #
###########################
include("checksimdata.jl")
include("func_forms.jl")
include("gendata.jl")
include("solver.jl")

"""
    DLWGMMIV.sim_data(N, T; <keyword arguments>)

Returns a `NamedTuple` containing a `DataFrame` of a panel dataset of `N` firms over `T+1` periods with specified parameters.

# Arguments
- `num_inputs::Integer=2`: the total number of production inputs to generate.
- `num_indp_inputs::Integer=1`: the number of independent inputs to generate.
- `input_names::Vector{String}`: a list of input names. Default is `["K","L"]. Additional inputs get a value of "X1","X2",... .
- `prod_params::Vector{Real}`: a list of parameters for the production function. Default is [0.1, 0.25]. Additional inputs get a value that is equal to 1 minus sum(prod_params) divided by the number of additional inputs, and TransLog second order terms get a value of 0.
- `cost_params::Vector{Real}`: a list of parameters for the cost function. Default is [0, 0.15]. Additional inputs get a value of 0.
- `omega_params::Vector{Real}`: a list of parameters for production technology function. Default is [0, 0.8, 0, 0].
- `indp_inputs_params:: Vector{Real}`: a list of parameters for independent inputs process. Default is [1]. Additional independent inputs get a value of 1.
- `σ_ω::Real=1`: the variance associated with the productivity shock each period.
- `indp_inputs_lnmean::Vector{Real}`: a list of natural log of mean values for each independent input. Default is [5]. Additional independent inputs get a value of 5.
- `indp_inputs_lnvariance::Vector{Real}`: a list of variances for each natural log of independent input. Default is [1]. Additional independent inputs get a value of 1.
- `seed::Integer`: sets a seed for `Random` number generator. Default is `-1`, no seed set.
- `X_start::Integer=1000`: set starting values for optimizer which calculates optimal level of dependent inputs for each firm.

# Configurable Options
- `prodF::String`: the production function parameter. Default is `"CD"`, Cobb-Douglas; other options include `"tl"`, TransLog.
- `costF::String`: the cost function parameter. Default is `"ce"`, constant elasticity.

# Examples
```jldoctest
julia> using DLWGMMIV

julia> df = DLWGMMIV.sim_data(20, 10).df
Sim Data for 2 inputs, CD

K Parameters:
    K_prod_params = 0.1 | K_cost_params = 0.0
L Parameters:
    L_prod_params = 0.25 | L_cost_params = 0.15

    First order derivative at optimal L is approximately zero: true

    Second order derivative at optimal L check: true

=======================

SUMMARY:
        100.0% of observations passed first order conditions.
        100.0% of observations passed second order conditions.

=======================
    
220×18 DataFrame
Row │ time   firm   S           Y           P          TC        omega_i    XI          K         L          C_K       C_L        rent_K   rent_L ⋯
    │ Int64  Int64  Float64     Float64     Float64    Float64   Float64    Float64     Float64   Any        Float64   Float64    Float64  Float6 ⋯
────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1 │     0      1   0.594649    3.60282    -647.634   651.237    0.687066   0.0        650.454   0.808584   650.454   0.783221       1.0  0.9686 ⋯
  2 │     0      2   0.151929    1.02321    -119.13    120.154   -0.12898    0.0        119.931   0.270617   119.931   0.222438       1.0  0.8219  
  3 │     0      3  -0.0790732   0.4036      -89.6328   90.0364  -0.828257   0.0         89.9487  0.120514    89.9487  0.0877392      1.0  0.7280  
 ⋮  │    ⋮      ⋮        ⋮           ⋮           ⋮          ⋮          ⋮          ⋮          ⋮          ⋮         ⋮          ⋮             ⋮        ⋮   ⋱ 
219 │    10     19  -0.26828     0.120173   -188.737   188.857   -1.85054    0.0152269  188.831   0.0420262  188.831   0.0261245      1.0  0.6216  
220 │    10     20  -0.475736    0.0675116   -83.0301   83.0976  -2.21972   -1.24907     83.0829  0.025454    83.0829  0.0146764      1.0  0.5765 ⋯                                                                                                                      
                                                                                                                      5 columns and 215 rows omitted

julia> df = DLWGMMIV.sim_data(20, 10, num_inputs = 3, input_names = ["k", "l", "m"], prod_params = [0.1, 0.25, 0.2, 0.05], prodF ="tl").df
Sim Data for 3 inputs, tl

k Parameters:
    k_prod_params = 0.1 | k_cost_params = 0.0  
l Parameters:
    l_prod_params = 0.25 | l_cost_params = 0.15
m Parameters:
    m_prod_params = 0.2 | m_cost_params = 0.0  
kl Parameters:
    kl_prod_params = 0.05 |
km Parameters:
    km_prod_params = 0.0 |
lm Parameters:
    lm_prod_params = 0.0 |
k2 Parameters:
    k2_prod_params = 0.0 |
l2 Parameters:
    l2_prod_params = 0.0 |
m2 Parameters:
    m2_prod_params = 0.0 |

******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
    Ipopt is released as open source code under the Eclipse Public License (EPL).
            For more information visit https://github.com/coin-or/Ipopt
******************************************************************************


    First order derivative at optimal L is approximately zero: true

    Second order derivative at optimal L check: true

=======================

SUMMARY:
        100.0% of observations passed optimization generating the simulated data.
        100.0% of observations passed first order conditions.
        100.0% of observations passed second order conditions.

=======================

220×24 DataFrame
Row │ time   firm   XI         k          l          m           Y          S          P          TC        omega_i    termination     C_k        ⋯   
    │ Int64  Int64  Float64    Float64    Float64    Float64     Float64    Float64    Float64    Float64   Float64    String          Float64    ⋯
────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  1 │     0      1   0.0       245.378    0.0790147  0.0236492   0.118246   -1.53147   -245.337   245.456   -0.603513  LOCALLY_SOLVED  245.378    ⋯ 
  2 │     0      2   0.0       194.093    1.1433     0.522569    2.61285     0.465793  -193.17    195.782    0.494648  LOCALLY_SOLVED  194.093     
  3 │     0      3   0.0        55.9995   0.0584379  0.0194533   0.0972665  -1.66692    -55.9599   56.0572  -0.663383  LOCALLY_SOLVED   55.9995    
 ⋮  │     ⋮      ⋮        ⋮          ⋮          ⋮          ⋮           ⋮          ⋮            ⋮         ⋮          ⋮             ⋮             ⋮      ⋱ 
219 │    10     19   0.117099  118.572    1.55853    0.783866    3.91933     0.64574   -117.102   121.021    0.72018   LOCALLY_SOLVED  118.572     
220 │    10     20   0.460817    6.10519  0.227785   0.123259    0.616295   -0.741436    -5.7946    6.4109   0.257407  LOCALLY_SOLVED    6.10519  ⋯                                                                                                                     
                                                                                                                     11 columns and 215 rows omitted                                                                                                                      
```
"""
function simfirmdata(
    N::Int, T::Int; 
    params::AbstractParameters = Parameters(),
    closed_solve::Bool = true,
    indp_inputs::AbstractMatrix{<:Real} = Matrix{Float64}(undef, N*T, 0),   
    omega::Vector{<:Real} = Vector{Float64}(undef, 0),
    Xomegas::AbstractMatrix{<:Real} = Matrix{Float64}(undef, N*T, 0),
    fixcost::Vector{<:Real} = Vector{Float64}(undef, 0),
    fixcosterror::Bool = false,
    xstart::Real = 1000,  
    opt_error::RealOrVec = 0,
    seed::Int = -1
    )

    if seed >= 0
        Random.seed!(seed)
    end

    df = init_df(N, T, params, indp_inputs, omega, Xomegas, fixcost, fixcosterror)
    df = solve_firm_decision(df, params, closed_solve, xstart, opt_error)

    return df
end
