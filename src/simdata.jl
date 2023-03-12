###########################
# Generate Simulated Data #
###########################
include("simdataCD.jl")
include("checksimdata.jl")


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
function sim_data(N, T; num_inputs = 2, num_indp_inputs = 1,  input_names = ["K", "L"], prod_params = [0.1, 0.25], cost_params = [0, 0.15], omega_params = [0, 0.8, 0, 0], indp_inputs_params = [1], σ_ω = 1, indp_inputs_lnmean = [5], indp_inputs_lnvariance = [1], seed = -1, X_start = 1000, prodF = "CD", costF = "ce")
    println("\n\nSim Data for $(num_inputs) inputs, $(prodF)")

    if prodF == "CD"
        return sim_data_CD(N, T, num_inputs = num_inputs, num_indp_inputs = num_indp_inputs,  input_names = input_names, prod_params = prod_params, cost_params = cost_params, omega_params = omega_params, indp_inputs_params = indp_inputs_params, σ_ω = σ_ω, indp_inputs_lnmean = indp_inputs_lnmean, indp_inputs_lnvariance = indp_inputs_lnvariance, seed = seed)
    end

    if seed >= 0
        Random.seed!(seed)
    end

    check_options(num_inputs, num_indp_inputs, prodF, costF)

    # Generate input names
    input_names = gen_input_names(num_inputs, input_names)
    
    # Set up parameters
    prod_params, cost_params, omega_params, indp_inputs_params = gen_params(num_inputs, num_indp_inputs, input_names, prodF, prod_params, cost_params, omega_params, indp_inputs_params)

    # Functions to Generate Data ##
    S_func, TC_func = gen_prodF_costF(prodF, costF, prod_params, cost_params, num_inputs)

    # Initialize independent variables, ω_it, and TFP shock distribution
    TFP_shock_dist = Normal(0, σ_ω)
    ω_it = rand(TFP_shock_dist, N)
    ξ = zeros(N)
    TFP_shock = ξ

    # Initialize independent inputs, e.g. k, capital
    x_indp = gen_indp_inputs(N, num_indp_inputs, indp_inputs_lnmean, indp_inputs_lnvariance)
    
    # Initialize dataframe
    firm_decision_df = DataFrame()

    ## Ipopt Model ##

    # Initialize model
    model = Model(Ipopt.Optimizer; add_bridges=false)
    set_optimizer_attribute(model, "bound_relax_factor", 0.0)
    set_silent(model)
    @NLparameter(model, p[i = 1:(num_indp_inputs+1)] == i)

    ## Set variables and NLexpressions
    @variable(model, Xdep[1:(num_inputs-num_indp_inputs)] >= 0, start = X_start)
    register(model, :S_func, num_inputs, S_func; autodiff=true)
    register(model, :TC_func, num_inputs, TC_func; autodiff=true)
    @NLexpression(model, S, S_func(p[1:num_indp_inputs]..., Xdep...))
    @NLexpression(model, Y, exp(S + p[end]))
    @NLexpression(model, TC, TC_func(p[1:num_indp_inputs]..., Xdep...))
    @NLexpression(model, P, Y - TC )

    ## defining objective
    @NLobjective(model, Max, P)

    #######################################
        
    for t in 0:T
        for n in 1:N
            for i in 1:num_indp_inputs
                set_value(p[i], x_indp[i][n])
            end
            set_value(p[2], ω_it[n])
            
            # Run firm decision model 
            JuMP.optimize!(model)  
            firm_decision = gen_firm_decision(model, TC_func, input_names)
            # Save results to dataframe
            push!(firm_decision_df, firm_decision; cols = :union)
        end
        
        # Save ξ (TFP shock)
        if t > 0
            ξ = [ξ; TFP_shock]
        end

        # Add periodic TFP shock for each firm
        TFP_shock = rand(TFP_shock_dist, N)
        ω_it .= [ω_it.^0 ω_it ω_it.^2 ω_it.^3]*omega_params .+ TFP_shock
        x_indp = [x_indp[i]*(indp_inputs_params[i]*rand(Normal(1,0.1))) for i in 1:num_indp_inputs]
    end

    # Save results to DataFrame
    firm_decision_df = hcat(DataFrame(time = repeat(0:T, inner = N), firm = repeat(1:N, outer = (T+1)), XI = ξ), firm_decision_df)

    # Store parameters and functions
    params = (prod_params = prod_params, cost_params = cost_params, omega_params = omega_params, indp_inputs_params = indp_inputs_params, σ_ω = σ_ω, prodF = prodF) 
    funcs = (S_func = S_func, TC_func = TC_func)
    input_params = (input_names = input_names, num_inputs = num_inputs, num_indp_inputs = num_indp_inputs)

    # Summary
    vc = sim_data_validity_check(firm_decision_df, params, funcs, input_params)
    total_obs = nrow(firm_decision_df)
    println("\n=======================\n")
    println("SUMMARY:")
    println("\t$(round(100*count(i->(i=="LOCALLY_SOLVED"),firm_decision_df.termination)/total_obs, digits = 2))% of observations passed optimization generating the simulated data.")
    println("\t$(round(100*count(vc.foc_pass)/total_obs, digits = 2))% of observations passed first order conditions.")
    println("\t$(round(100*count(vc.soc_pass)/total_obs, digits = 2))% of observations passed second order conditions.")
    println("\n=======================\n")
        
    return (df=firm_decision_df, params=params, funcs = funcs, input_params = input_params, derivative_check = vc)
end

function check_options(num_inputs, num_indp_inputs, prodF, costF)
    prodF_options = ["CD", "tl"]
    costF_options = ["ce"]
    if (num_inputs < 2) || (num_indp_inputs >= num_inputs) || (prodF ∉ prodF_options) || (costF ∉ costF_options)
        #TO DO
        println("error!!")
        return
    end

    return prodF_options, costF_options
end

function gen_input_names(num_inputs, input_names)
    if length(input_names) < num_inputs
        for j in (length(input_names)+1):num_inputs
            input_names = [input_names; "X"*string(j)]
        end
    else
        input_names = input_names[begin:num_inputs]
    end

    return input_names
end

function gen_prodF_costF(prodF, costF, prod_params, cost_params, num_inputs)
    ### Production Functions
    funcs = []
    if prodF == "CD"
        # Cobb Douglas
        CobbDouglas(x...) = sum(log(x[i])*prod_params[i] for i in eachindex(x))
        funcs = [funcs; CobbDouglas]
    elseif prodF == "tl"
        # TransLog
        num_tl_terms = length(prod_params)
        TL_interactions(prod_params, x...) = sum([log(x[xcomb[1]])*log(x[xcomb[2]]) for xcomb in combinations(eachindex(x),2)][i]*prod_params[i] for i in eachindex(prod_params) )
        TransLog(x...) = sum(log(x[i])*prod_params[i] + (log(x[i])^2)*prod_params[(num_tl_terms-num_inputs+i)] for i in eachindex(x)) + TL_interactions(prod_params[(num_inputs+1):(num_tl_terms-num_inputs)], x...)
        funcs = [funcs; TransLog]
    end
 
    ### Cost Functions
    if costF == "ce"
        TC_1(x...) = sum(x[i]^(1+cost_params[i]) for i in eachindex(x))
        funcs = [funcs; TC_1]
    end

    return funcs
end

function gen_params(num_inputs, num_indp_inputs, input_names, prodF, prod_params, cost_params, omega_params, indp_inputs_params)
    vars = input_names
    num_prod_params = length(prod_params)
    num_cost_params = length(cost_params)
    num_omega_params = length(omega_params)
    num_indp_inputs_params = length(indp_inputs_params)
    
    sum_prod_params = sum(prod_params)
    prod_params = num_prod_params < num_inputs ? [prod_params; repeat([(1-sum_prod_params)/(num_inputs-num_prod_params)], (num_inputs-num_prod_params))] : prod_params
    cost_params = num_cost_params < num_inputs ? [cost_params; zeros(num_inputs-num_cost_params)] : cost_params[begin:num_inputs]
    if num_omega_params != 4  
        println("error!") # TODO
    end
    indp_inputs_params = num_indp_inputs_params < num_indp_inputs ? [indp_inputs_params; ones(num_indp_inputs-num_indp_inputs_params)] : indp_inputs_params[begin:num_indp_inputs]

    num_prod_params = length(prod_params)
    if prodF == "CD"
        prod_params = prod_params[begin:num_inputs]
    elseif prodF == "tl"
        vars = [input_names; [join(xcomb) for xcomb in combinations(input_names,2)] ; input_names.*"2"] 
        num_tl_terms = num_inputs*2 + binomial(num_inputs,2) 
        prod_params = num_prod_params < num_tl_terms ? [prod_params; zeros(num_tl_terms-num_prod_params)] : prod_params[begin:num_tl_terms]
    end

    for i in eachindex(vars)
        println("\n$(vars[i]) Parameters:")
        print("  $(vars[i])_prod_params = $(prod_params[i]) | ")
        if i <=num_inputs
            print("$(vars[i])_cost_params = $(cost_params[i])")
        end
    end
    println("")
    
    return prod_params, cost_params, omega_params, indp_inputs_params
end

function gen_indp_inputs(num_samples, num_indp_inputs, indp_inputs_lnmean, indp_inputs_lnvariance)
    indp_inputs_lnmean = length(indp_inputs_lnmean) < num_indp_inputs ? [indp_inputs_lnmean; 5*ones(num_indp_inputs - length(indp_inputs_lnmean))] : indp_inputs_lnmean[begin:num_indp_inputs]
    indp_inputs_lnvariance = length(indp_inputs_lnvariance) < num_indp_inputs ? [indp_inputs_lnvariance; ones(num_indp_inputs - length(indp_inputs_lnvariance))] : indp_inputs_lnvariance[begin:num_indp_inputs]
    indp_inputs_distributions = MvLogNormal(indp_inputs_lnmean, diagm(indp_inputs_lnvariance))
    X_indp = rand(indp_inputs_distributions, num_samples)
    x_indp = [X_indp[i,:] for i in 1:num_indp_inputs]

    return x_indp
end

function gen_firm_decision(model, TC_func, input_names)
    X_indp = value.(model[:p][1:(end-1)]) 
    X = [X_indp; value.(model[:Xdep])]
    X_opt = (; zip(Symbol.(input_names), X)...)
    outcomes = ( Y = value(model[:Y]), S= value(model[:S]), P = value(model[:P]), TC = value(model[:TC]), omega_i = value(model[:p][end]), termination = string(termination_status(model)) ) 
    other_outcomes = other_firm_outcomes(TC_func, X, outcomes.TC, outcomes.Y, input_names)
    
    firm_decision = merge(X_opt, outcomes)
    for outcome in other_outcomes
        firm_decision = merge(firm_decision, outcome)
    end

    return firm_decision
end

function other_firm_outcomes(TC_func, X, TC, Y, input_names)
    input = zeros(length(X))
    C = []
    rent = []
    for i in eachindex(X)
        input[i] = X[i]
        C = [C; TC_func(input...)]
        rent = [rent; C[i]/X[i] ]
        input[i] = 0
    end
    share_TC = C ./ TC
    share_Y = C ./ Y

    C_res = (; zip(Symbol.("C_".*input_names), C)...)
    rent_res = (; zip(Symbol.("rent_".*input_names), rent)...)
    share_TC_res = (; zip(Symbol.("share_TC_".*input_names), share_TC)...)
    share_Y_res = (; zip(Symbol.("share_Y_".*input_names), share_Y)...)

    return C_res, rent_res, share_TC_res, share_Y_res
end
