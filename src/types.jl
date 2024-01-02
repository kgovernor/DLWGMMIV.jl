abstract type AbstractGMMIVModel end
abstract type AbstractResults end
abstract type AbstractBetas end
abstract type AbstractParameters end
abstract type AbstractFunctions <: AbstractParameters end
RealOrVec = Union{T, Vector{T}} where {T<:Real}
RealOrVecReal = Union{T, Vector{<:RealOrVec{T}}} where {T<:Real}
BetaOrVec = Union{T, Vector{<:Real}} where {T<:AbstractBetas}

mutable struct Betas{B<:AbstractDict{String, Vector{Float64}}} <: AbstractBetas
    b::B
    deg::Int
    e::Vector{Vector{Int}}
end

struct GMM_results{T1, T2} <: AbstractResults
    beta_dlw::T1
    sol::T2
    globalsolve::Bool
    successful::Bool
    vstart::Float64 
    vend::Float64 
    vstart_inner::Float64 
    vend_inner::Float64
end

struct GMM_model{B <: AbstractBetas} <: AbstractGMMIVModel
    method::Symbol
    s1_deg::Int
    s2_deg::Int
    g_B::B
    omega_deg::Int
    use_constant::Bool 
    gOLS::Bool
end

struct FirmSolution <: AbstractResults
    Y::Vector{Float64} 
    X::AbstractMatrix{Float64}
    termination::Vector{String}
end

struct Functional_Forms{T<:AbstractDict{Symbol, Function}} <: AbstractFunctions
    prod::T
    cost::T
end

struct ParamsIndex <: AbstractParameters
    coefs::String
    exps::String
    start::String
    sub::String
    deg_homog::String
end

struct Params{B <: AbstractBetas, D <: Distribution} <: AbstractParameters
    func::Symbol
    params::OrderedDict{String, B}
    err::D
    i::ParamsIndex
end

struct SetParameters<: AbstractParameters
    n::Int
    n_indp::Int
    prod_params::Params
    cost_params::Params 
    fixcost_params::Params
    omega_params::Params
    omega_inputs_params::Params
    indp_params::Params
end

mutable struct Cache <: AbstractResults
    β::Any
    v::Any
    g_β::Any
end

function GMM_results(beta_dlw, sol, globalsolve, successful, vstart, vend, vstart_inner, vend_inner)
    return GMM_results(beta_dlw, sol, globalsolve, successful, vstart, vend, vstart_inner, vend_inner)
end

function GMM_model(
    method = :acf,
    stage1_deg = 3,
    stage2_deg = 2,
    omega_deg = 3,
    use_constant = true,
    gOLS = false,
    g_B = Betas(["ω"], deg = omega_deg, constant = true, init = false)
    )
    return GMM_model(method, stage1_deg, stage2_deg, g_B, omega_deg, use_constant, gOLS)
end

ACF_model(;stage1_deg = 3, stage2_deg = 2, omega_deg = 3, use_constant = true, gOLS = true) = GMM_model(:acf, stage1_deg, stage2_deg, omega_deg, use_constant, gOLS)

function VectorVectorFloat(x::RealOrVecReal, len::Int = 1)
    V = Vector{Vector{Float64}}(undef, 0)
    if x isa Vector
        for i in eachindex(x)
            if x[i] isa Vector
                push!(V, Float64.(x[i]))
            else
                push!(V, [Float64(x[i])])
            end
        end
    else
        for _ in 1:len
            push!(V, [Float64(x)])
        end
    end
    return V
end

function Betas(b::AbstractDict, deg::Int, e::Vector)
    return Betas(b, deg, e)
end

function Betas(vars::Int = 0, initialvalues::RealOrVecReal = 0; deg::Int = 1, constant::Bool = false, init::Bool = true)
    if vars > 0 
        V = Vector{String}(undef, vars)
        for i in eachindex(V)
            V[i] = "β$i"
        end
    else
        V = Vector{String}(undef, 0)
    end

    return Betas(V, initialvalues, deg = deg, constant = constant, init = init)
end

function Betas(vars::Vector{String} = Vector{String}(undef, 0), initialvalues::RealOrVecReal = 0; deg::Int = 1, constant::Bool = false, init::Bool = true)
    betas = OrderedDict{String, Vector{Float64}}()

    pv = poly_vars(vars, deg)
    vars = [vars; pv.v]
    
    if constant
        vars = ["0"; vars] 
    end 

    for v in vars
        betas[v] = Vector{Float64}(undef, 0)
    end

    initialvalues = VectorVectorFloat(initialvalues, length(vars))

    B = Betas(betas, deg, Vector{Vector{Int}}(pv.e))

    init = isempty(vars) ? false : init
    if init
        start_betas!(B, vars, initialvalues)
    end

    return B
end

function mod_betas(B::AbstractBetas, betas::Vector{String}, values::RealOrVecReal; add::Bool = false, replace::Bool = false)
    values = VectorVectorFloat(values)
    if length(betas) == length(values)
        B = replace ? B : deepcopy(B)   
        for i in 1:length(betas)
            b = betas[i]
            if b in keys(B)
                if !add
                    B[b] = []
                end
                B[b] = [B[b]; values[i]]
            else
                B[b] = [[]; values[i]]
            end
        end
    else
        throw(DimensionMismatch("Size of betas not equal to size of start_values; got a dimension with lengths $(length(betas)) and $(length(values))"))
    end

    return B
end

start_betas(B, betas, values) = mod_betas(B, betas, values)
add_betas(B, betas, values) = mod_betas(B, betas, values, add = true)
start_betas!(B, betas, values) = mod_betas(B, betas, values, replace = true)
add_betas!(B, betas, values) = mod_betas(B, betas, values, add = true, replace = true)

###############################################

function Params(params = OrderedDict{String, AbstractBetas}(); func = :none, err = Normal(0, 0), i = ParamsIndex())
    return Params(func, params, err, i)
end

function ParamsIndex(
    coefs = "coefs", 
    exps = "exps", 
    start = "start",
    sub = "sub",
    deg_homog = "deg_homog"
)
    return ParamsIndex(coefs, exps, start, sub, deg_homog)
end

function SetParameters(n, n_indp, prod_params, cost_params, omega_params, indp_params)
    return SetParameters(n, n_indp, prod_params, cost_params, omega_params, indp_params)
end
function Parameters(;
    num_inputs::Int = 2,
    num_indp::Int = 1,

    prod_func::Symbol = :cd,
    cost_func::Symbol = :add_seperable,

    prod_coefs::BetaOrVec = [1/num_inputs for _ in 1:num_inputs],
    cost_coefs::BetaOrVec = ones(num_inputs),
    cost_exps::BetaOrVec = ones(num_inputs),

    fixcost_coefs::Vector{<:Real} = [0, 1],
    fixcost_start::Real = 0,
    omega_coefs::Vector{<:Real} = [0, 0.8],
    omega_start::Real = 0,
    omega_inputs_coefs::RealOrVecReal = 0,
    omega_inputs_start::RealOrVec = zeros(num_inputs),
    indp_coefs::RealOrVecReal = 1,
    indp_start::RealOrVec = ones(num_indp),

    sub::Real = 0.5,
    deg_homog::Real = 1,

    err_dist_output = Normal(0, 0),
    err_dist_fixcost = LogNormal(0, 0),
    err_dist_omega = Normal(0, 1),
    err_dist_omega_inputs = MvNormal(zeros(num_inputs), 0),
    err_dist_indp = MvLogNormal(zeros(num_indp), I)
    )
    
    if num_inputs < num_indp
        throw(ArgumentError("number of inputs $num_inputs is less than number of independent inputs $num_indp"))
    end
    funcs = Functional_Forms()
    check_func(keys(funcs.prod), prod_func)
    check_func(keys(funcs.cost), cost_func)

    p = ParamsIndex()
    prod_exps = Betas(
        [p.sub, p.deg_homog], 
        [[sub], [deg_homog]]
    )

    if prod_coefs isa Vector
        prod_coefs = prod_func == :tl ? Betas(num_inputs, prod_coefs, deg = 2) : Betas(num_inputs, prod_coefs)
    end
    if cost_coefs isa Vector
        cost_coefs = cost_func == :tl ? Betas(num_inputs, cost_coefs, deg = 2) : Betas(num_inputs, cost_coefs)
    end
    if cost_exps isa Vector
        cost_exps = cost_func == :tl ? Betas(num_inputs, cost_exps, deg = 2) : Betas(num_inputs, cost_exps)
    end
    
    prod_params = Params( 
        OrderedDict{String, AbstractBetas}(
            p.coefs => prod_coefs, 
            p.exps => prod_exps
        ),
        func = prod_func,
        err = err_dist_output
    )
    cost_params = Params(
        OrderedDict{String, AbstractBetas}(
            p.coefs => cost_coefs,
            p.exps => cost_exps
        ),
        func = cost_func
    )
    fixcost_params = Params(
        OrderedDict{String, AbstractBetas}(
            p.start => Betas([p.start], fixcost_start),
            p.coefs => Betas(1, [fixcost_coefs])
        ),
        err = err_dist_fixcost
    )
    omega_params = Params(
        OrderedDict{String, AbstractBetas}(
            p.start => Betas([p.start], omega_start),
            p.coefs => Betas(1, [omega_coefs])
        ),
        err = err_dist_omega
    )
    omega_inputs_params = Params(
        OrderedDict{String, AbstractBetas}(
            p.start => Betas([p.start], [omega_inputs_start]), 
            p.coefs => Betas(num_inputs, omega_inputs_coefs)
        ),
        err = err_dist_omega_inputs
    )
    indp_params = Params(
        OrderedDict{String, AbstractBetas}(
            p.start => Betas([p.start], [indp_start]), 
            p.coefs => Betas(num_indp, indp_coefs)
        ),
        err = err_dist_indp
    )

    return SetParameters(
        num_inputs, 
        num_indp,
        prod_params,
        cost_params,
        fixcost_params,
        omega_params,
        omega_inputs_params,
        indp_params
        )    
end

###########################


function FirmSolution(Y_opt::Vector{Real} = Vector{Real}(undef, 0), X_opt::AbstractMatrix{Real} = Matrix{Real}(undef, 0, 0), termination::Vector{String} = Vector{String}(undef, 0))
    return FirmSolution(Y_opt, X_opt, termination)
end

#############################

function Functional_Forms()
    prod = Dict{Symbol, Function}(
        :cd => CobbDouglas, 
        :ces => CES, 
        :ves => VES, 
        :tl => Translog
    )
    cost = Dict{Symbol, Function}(
        :add_seperable => AddSeperable
    )    
    return Functional_Forms(prod, cost)
end

    
