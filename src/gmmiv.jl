

################
# Main methods #
################


"""
    dlwGMMIV(year, plantid, Q, inputs...; <keyword arguments>)

Returns results for estimated betas from GMM IV for industry's production function `prodF` given panel data with `year`, `plantid`, `inputs...` (with independent/dynamic inputs listed first), and output, `Q`.
 
Results are returned as a `NamedTuple`.

# Arguments
- `num_indp_inputs::Integer=1`: the number of independent/dynamic inputs.
- `bstart::Vector{Float64}`: the starting beta values for optimization. Defaults to zeros.
- `δ_nm::Float64`=0.1: the Nelder-Mead shrink step parameter, delta.
- `max_iters::Integer=500`: the maximum number of iterations ran in optimization.

# Configurable Options
- `prodF::String`: the production function to estimate. Default is `"CD"`, Cobb-Douglas; other options include `"tl"`, TransLog.
- `opt::String`: the optimization method to use. Default is `"nm"`, Nelder-Mead; other options include `"LBFGS"`.

# Examples
```jldoctest
julia> using DLWGMMIV

julia> df = DLWGMMIV.sim_data(20000, 10, input_names = ["K", "L"], prod_params = [0.1, 0.25]).df;
[...]

julia> data = [df.time, df.firm, df.Y, df.K, df.L];

julia> res = dlwGMMIV(data...);

julia> println("
Converge = \$(res.conv_msg)\\n
Objective Value = \$(res.other_results.crit)\\n
betas = \$(res.beta_dlw)
")
Converge = true

Objective Value = 1.074768000861536e-9

betas = (beta_x1 = 0.10230132568110045, beta_x2 = 0.26251910427769287)

julia> res = dlwGMMIV(data..., opt = "LBFGS");

julia> println("
Converge = \$(res.conv_msg)\\n
Objective Value = \$(res.other_results.crit)\\n
betas = \$(res.beta_dlw)
")
Converge = true

Objective Value = 3.6354547893764317e-19

betas = (beta_x1 = 0.10230132472854156, beta_x2 = 0.2625190996231563)

julia> res = dlwGMMIV(data..., opt = "LBFGS", prodF = "tl");

julia> println("
Converge = \$(res.conv_msg)\\n
Objective Value = \$(res.other_results.crit)\\n
betas = \$(res.beta_dlw)
")
Converge = true

Objective Value = 3.1089429872638027e-15

betas = (beta_x1 = 0.11394854106849515, beta_x2 = 1.9222327345314836, beta_x1x2 = -0.5175358152780514, beta_x12 = -0.04266689698177511, beta_x22 = 0.00703394572086843)
```
"""
function dlwGMMIV(
    df::DataFrame, by::Vector{String}, y::Vector{String}, x_stage1::Vector{String}, x_stage2::Vector{String}, z::Vector{String};  
    betas = Betas(),
    model = ACF_model(),
    skip_stage1 = false,
    bstart = 0,
    globalsolve = false,
    local_optimizer = BFGS(),
    global_optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited(),
    maxtime = 1000.0,
    maxiters = 20000,
    lb = 0,
    ub = 1,
    df_out = false
    )

    vars = (
        by = by,
        yvar = y[1],
        xvars1 = x_stage1,
        xvars2 = x_stage2,
        zvars = z,
        phivar = "phi", 
        epsvar = "epsilon"
    )

    opt = (
        bstart = bstart, 
        globalsolve = globalsolve,
        local_optimizer = local_optimizer,
        global_optimizer = global_optimizer,
        maxtime = maxtime,
        maxiters = maxiters,
        lb = lb,
        ub = ub    
    )

    df = deepcopy(df)
        
    s1_res = stage1(df, vars, model, skip_stage1)
    s2_res = stage2(df, vars, betas, model, opt)

    res = (s1 = s1_res, s2 = s2_res)

    output = df_out ? (r = res, df = df) : (r = res)
    return output
end

function dlwGMMIV(time::Vector, id::Vector, Y::Vector, X_stage1::AbstractMatrix, X_stage2::AbstractMatrix, Z::AbstractMatrix;
    betas = Betas(),
    model = ACF_model(),
    skip_stage1 = false,
    bstart = 0,
    globalsolve = false,
    local_optimizer = BFGS(),
    global_optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited(),
    maxtime = 1000.0,
    maxiters = 20000,
    lb = 0,
    ub = 1,
    df_out = false
    ) 

    vars = init_vars(size(X_stage1, 2), size(X_stage2, 2), size(Z, 2))
    df = init_df(time, id, Y, X_stage1, X_stage2, Z, vars)

    return dlwGMMIV(df, vars.by, vars.y, vars.xvars1, vars.xvars2, vars.z,  
        betas = betas,
        model = model,
        skip_stage1 = skip_stage1,
        bstart = bstart, 
        globalsolve = globalsolve,
        local_optimizer = local_optimizer,
        global_optimizer = global_optimizer,
        maxtime = maxtime,
        maxiters = maxiters,
        lb = lb,
        ub = ub, 
        df_out = df_out
    )
end

################################################################################################################################################
#################
# Other methods #
#################

function stage1(df, vars, model, skip_stage1)
    res = missing

    df[!, vars.phivar] = df[!, vars.yvar]
    
    if model.method == :none || skip_stage1
        println("\nSkipping Stage 1. Returning output as phi.\n")
    else 
        y = df[!, vars.yvar]
        X = df[!, vars.xvars1]

        if model.method in [:acf]
            println("\nStage 1 polynomial regression approximation. Returning output as phi.\n")
            res = mv_polyreg(y, X, model.s1_deg)
            df[!, vars.phivar] = predict(res)
        else
            throw(ArgumentError("$(model.method) is not a valid stage 1 estimation method."))
        end
    end

    df[!, vars.epsvar] = df[:, vars.yvar] - df.phi 

    return res
end

function stage2(df, vars, betas, model, opt)
    B = init_betas(betas, "β_".*(vars.xvars2), opt.bstart, model.s2_deg)
    nB = length(B)
    g_B = model.g_B
    ngB = length(g_B)

    for v in [vars.xvars2, vars.zvars]
        df = poly_approx_vars(df, B.deg, v, exponents = B.e)
    end    

    vars = merge(
        vars,
        (xvars2 = [vars.xvars2; poly_vars(vars.xvars2, B.deg, exponents = B.e).v],
        zvars = [vars.zvars; poly_vars(vars.zvars, B.deg, exponents = B.e).v])
    )
    Z = Matrix(df[completecases(df), vars.zvars])
    ZZ = Z * Z'

    prodest = prodest_method(model.method)
    f(betas, gbetas; derivative = []) = prodest(betas, gbetas, df, vars, g_B, derivative = derivative)
    gf(betas) = prodest(betas, df, vars, g_B, model.use_constant)

    function solve_lower_level(betas)
        if opt.global_optimizer == :IntervalOptimiser
            betas = mid.(betas)
        end
        innergmm(gbetas) = sum(f(betas, gbetas).^2)
        if model.gOLS
            gbs = gf(betas) 
            v = innergmm(gbs)
        else
            if model.use_constant
                res = gmm(innergmm, ngB, lb = zeros(ngB) .- eps(), ub = ones(ngB))
                gbs = sol.u
            else
                res = gmm((b) -> innergmm([0; b]), ngB-1, lb = zeros(ngB-1) .- eps(), ub = ones(ngB-1))
                gbs = [0; sol.u]
            end
            @assert SciMLBase.successful_retcode(sol)
            v = sol.objective
        end
        
        return (v=v, gbs=gbs)
    end
    function _update_if_needed(cache::Cache, x)
        if cache.β !== x
            res = solve_lower_level(x)
            cache.v, cache.g_β = res
            cache.β = x
        end
        return
    end    
    function solveXI(cache::Cache, betas)
        _update_if_needed(cache, betas)
        XI = f(betas, cache.g_β)
        v = (XI' * ZZ * XI) / (nrow(df)^2)
        return v
    end
    cache = Cache([], [], [])

    lb, ub = zeros(nB) .- eps(), ones(nB)
    bstart = initial_values(B)

    optf = OptimizationFunction((bs, p) -> solveXI(cache, bs), Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, bstart, lb = lb, ub = ub)

    if opt.globalsolve
        if opt.global_optimizer isa Optim.ParticleSwarm
            sol = solve(prob, Optim.ParticleSwarm(lower = prob.lb, upper = prob.ub, n_particles = opt.global_optimizer.n_particles), maxiters = opt.maxiters, maxtime = opt.maxtime)
        elseif opt.global_optimizer == :IntervalOptimiser
            X = IntervalBox(0..1, nB)
            zmin, xmin = minimise((B) -> solveXI(cache, [b for b in B]), X, tol = 1e-2)
            println(xmin)
            return (z = zmin, x = xmin)
        else
            sol = solve(prob, opt.global_optimizer, maxiters = opt.maxiters, maxtime = opt.maxtime)
        end
    else
        sol = solve(prob, opt.local_optimizer, maxiters = opt.maxiters, maxtime = opt.maxtime)
    end
    # println(sol.original)

    println(sol.u)
    # println(sol.objective)

    add_betas!(B, keys(B), sol.u)
    innervstart, gbs = solve_lower_level(bstart)
    start_betas!(g_B, keys(g_B), gbs)
    innervend, gbs = solve_lower_level(sol.u)
    add_betas!(g_B, keys(g_B), gbs)

    res = GMM_results(
        [B, g_B],
        sol,
        opt.globalsolve,
        SciMLBase.successful_retcode(sol),
        solveXI(cache, bstart),
        sol.objective,
        innervstart,
        innervend
    )
    df = df[completecases(df), :]

    return res
end

function gmm(f, num_betas = 0; start = zeros(num_betas), lb = repeat([-Inf], num_betas), ub = repeat([Inf], num_betas))

    optf = OptimizationFunction((x, p) -> f(x), Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, start, lb = lb, ub = ub)

    sol = solve(prob, BFGS(), maxiters = 20000, maxtime = 1000.0)

    return sol
end

function mv_polyreg(y, X, deg)
    if deg > -1
        X = isa(X, Matrix) ? DataFrame(X, :auto) : X
        df = poly_approx_vars(X, deg)
        df[!, :y] = y
        reg = lm(term(:y) ~ sum(term.(names(df, Not("y")))), df)
    else
        throw(DomainError("$deg, an invalid degree for polynomial regression. deg must be a non-negative integer"))
    end

    return reg
end

function init_betas(B, names, bstart, deg)
    n = length(names)
    B.deg = deg

    if isempty(B)
        B = Betas(names, bstart, deg = deg)
    end

    num_terms = n
    for i in 2:B.deg
        num_terms += factorial(i+n-1)/(factorial(n-1)*factorial(i))
    end
    length(B) == num_terms || throw(ArgumentError("bstart does not have enough values for $n terms of degree $(B.deg); expecting $num_terms beta values"))

    return B
end

function init_df(time, id, Y, X_stage1, X_stage2, Z, vars)
    df = DataFrame([time id Y], [vars.by; vars.y])
    for i in eachindex(vars.xvars1)
        df[!, vars.xvars1[i]] = X_stage1[:, i]
    end
    for i in eachindex(vars.xvars2)
        df[!, vars.xvars2[i]] = X_stage2[:, i]
    end
    for i in eachindex(vars.zvars)
        df[!, vars.zvars[i]] = Z[:, i]
    end

    return df
end

function init_vars(len_x_s1, len_x_s2, len_z)
    vars = (
        by = ["time", "id"],
        yvar = "y",
        xvars1 = genvarnames("xs", len_x_s1),
        xvars2 = genxvars(len_x_s2),
        zvars = genvarnames("z", len_z),
    )
    return vars
end

function prodest_method(model)
    prod_est = Dict{Symbol, Function}(
        :acf => ACF, 
        # :gnr => GNR
    )
      
    return prod_est[model]
end

