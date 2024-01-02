Base.keys(B::Betas) = collect(keys(B.b))
Base.values(B::Betas, makevec::Bool = false) = makevec ? reduce(vcat, values(B.b)) : collect(values(B.b))
Base.getindex(B::Betas, b::String) = B.b[b]
Base.setindex!(B::Betas, v, b::String) = (B.b[b] = v)
Base.isempty(B::Betas) = isempty(B.b)
Base.length(B::Betas) = length(B.b)
function i_values(index, B::Betas) 
    V = Vector{Float64}(undef, length(B))
    b = values(B)
    for i in eachindex(b)
        V[i] = index == -1 ? b[i][end] : b[i][index]
    end
    return V
end
initial_values(B::Betas) = i_values(1, B)
final_values(B::Betas) = i_values(-1, B)

Base.keys(P::Params) = keys(P.params)
Base.values(P::Params, param::String, makevec::Bool = true) = makevec ? values(P[param], makevec) : values(P[param])
Base.getindex(P::Params, param::String) = P.params[param]
Base.isempty(P::Params) = isempty(P.params)

getnuminputs(P::SetParameters) = (P.n, P.n_indp)