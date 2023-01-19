module DLWGMMIV

# Write your package code here.
using Distributions
using Random
using Statistics
using LinearAlgebra
using DataFrames
using DataFramesMeta
using CSV
using JuMP
using Ipopt
using ForwardDiff
using Zygote


include("simdata.jl")

export sim_data_CD, sim_data_solved_L

end
