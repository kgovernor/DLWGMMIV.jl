###########################
# Generate Simulated Data #
###########################
include("simdataCD_KL.jl")
include("checksimdata.jl")

"""
    sim_data(N, T; num_inputs = 2, num_indp_inputs = 1,  input_names = ["k", "l"], prod_params = [0.1, 0.25], cost_params = [0, 0.15], omega_params = [0, 0.8, 0, 0], σ_ω = 1, seed = -1, start_X = 1000, prodF = "CD", costF = "ce")


TBW

"""
function sim_data(N, T; num_inputs = 2, num_indp_inputs = 1,  input_names = ["k", "l"], prod_params = [0.1, 0.25], cost_params = [0, 0.15], omega_params = [0, 0.8, 0, 0], σ_ω = 1, seed = -1, X_start = 1000, prodF = "CD", costF = "ce")
    println("\n\nSim Data for $(num_inputs) inputs, $(prodF)\n\n")
    if seed >= 0
        Random.seed!(seed)
    end

    check_options(num_inputs, num_indp_inputs, prodF, costF)

    # Generate input names
    input_names = gen_input_names(num_inputs, input_names)
    
    # Set up parameters
    prod_params, cost_params, omega_params = gen_params(num_inputs, input_names, prodF, prod_params, cost_params, omega_params)

    # Functions to Generate Data ##
    S_func, TC_func = gen_prodF_costF(prodF, costF, prod_params, cost_params, num_inputs)

    ## Ipopt Model ##

    # Initialize dataframe
    firm_decision_df = DataFrame()

    # Initialize model
    model = Model(Ipopt.Optimizer)
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
        
    # Initialize ω_it and TFP shock distribution
    ω_it = rand(Normal(0, 1), N)
    TFP_shock_dist = Normal(0, σ_ω)
    ξ = zeros(N)
    TFP_shock = ξ

    for t in 0:T
        # Initialize independent inputs, e.g. k, capital
        x_indp = [exp.(rand(Normal(0,1), N)) for i in 1:num_indp_inputs]

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
        ω_it .= [ones(length(ω_it)) ω_it ω_it.^2 ω_it.^3]*omega_params .+ TFP_shock
    end

    # Save results to DataFrame
    firm_decision_df = hcat(DataFrame(time = repeat(0:T, inner = N), firm = repeat(1:N, outer = (T+1)), XI = ξ), firm_decision_df)

    # Store parameters and functions
    params = (prod_params = prod_params, cost_params = cost_params, omega_params = omega_params, σ_ω = σ_ω, prodF = prodF) 
    funcs = (S_func = S_func, TC_func = TC_func)
    input_params = (input_names = input_names, num_inputs = num_inputs, num_indp_inputs = num_indp_inputs)

    return (df=firm_decision_df, params=params, funcs = funcs, input_params = input_params)
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
            input_names = [input_names; "x"*string(j)]
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

function gen_params(num_inputs, input_names, prodF, prod_params, cost_params, omega_params)
    vars = input_names
    num_prod_params = length(prod_params)
    num_cost_params = length(cost_params)
    num_omega_params = length(omega_params)
    
    sum_prod_params = sum(prod_params)
    prod_params = num_prod_params < num_inputs ? [prod_params; repeat([(1-sum_prod_params)/(num_inputs-num_prod_params)], (num_inputs-num_prod_params))] : prod_params
    cost_params = num_cost_params < num_inputs ? [cost_params; zeros(num_inputs-num_cost_params)] : cost_params[begin:num_inputs]
    omega_params = num_omega_params < 4 ? [omega_params; rand(4-num_omega_params)] : omega_params[begin:4]

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
    
    return prod_params, cost_params, omega_params
end

function gen_firm_decision(model, TC_func, input_names)
    X_indp = value.(model[:p][1:(end-1)]) 
    X = [X_indp; value.(model[:Xdep])]
    X_opt = (; zip(Symbol.(input_names), X)...)
    outcomes = ( Y = value(model[:Y]), S= value(model[:S]), P = value(model[:P]), TC = value(model[:TC]), omega_i = value(model[:p][end]), termination = string(termination_status(model)) ) 
    other_outcomes = other_firm_outcomes(TC_func, X, outcomes.TC, outcomes.Y, input_names)
    
    firm_decision = merge(X_opt, outcomes)
    for outcomes in other_outcomes
        merge(firm_decision, outcomes)
    end

    return firm_decision
end

function other_firm_outcomes(TC_func, X, TC, Y, input_names)
    other_firm_outcomes = ()
    input = zeros(length(X))
    C = []
    rent = []
    for i in eachindex(X)
        input[i] = X[i]
        C = [C; TC_func(input...)]
        rent = [rent; C[i]/X[i] ]
        X[i] = 0
    end
    share_TC = C ./ TC
    share_Y = C ./ Y

    C_res = (; zip(Symbol.("C_".*input_names), C)...)
    rent_res = (; zip(Symbol.("rent_".*input_names), rent)...)
    share_TC_res = (; zip(Symbol.("share_TC_".*input_names), share_TC)...)
    share_Y_res = (; zip(Symbol.("share_Y_".*input_names), share_Y)...)

    return C_res, rent_res, share_TC_res, share_Y_res
end
