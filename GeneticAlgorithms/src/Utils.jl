# src/utils.jl
module Utils

export random_population, clip, EAresult, get_experiments_file, random_indexes

function random_population(lower::Float64, upper::Float64, dimension::Int, size::Int)
    @assert lower < upper "Lower bound must be less than upper bound"
    population = lower .+ (upper .- lower) .* rand(dimension, size)
    return population
end

# Recorta la solución entre los valores del dominio
function clip(domain, solution)
    lower, upper = domain
    @assert lower < upper "Lower bound must be menos que upper bound"
    return clamp.(solution, lower, upper)
end

# Estructura para almacenar el resultado del algoritmo de evolución
# mutable struct EAresult
#     fitness::Float64
#     solution::Vector{Float64}
#     evaluations::Int
# end

# Obtiene el archivo de experimentos
# function get_experiments_file(name_output::Union{Nothing, String}, replace::Bool=false, times::Int=1)
#     if isnothing(name_output)
#         return nothing, nothing
#     else
#         if !replace && isfile(name_output)
#             lines = readlines(name_output)
#             if length(lines) >= times
#                 bestSolutionFitness, bestSol, bestEval, evaluations = split(lines[end], ',')
#                 bestSolutionFitness = parse(Float64, bestSolutionFitness)
#                 bestSol = parse.(Float64, split(bestSol))
#                 evaluations = parse(Int, evaluations)
#                 return EAresult(bestSolutionFitness, bestSol, evaluations), nothing
#             end
#         end
# 
#         fid = open(name_output, replace ? "w" : "a")
#         return nothing, fid
#     end
# end

# Retorna un grupo de índices aleatorios entre 0 y size, evitando los índices ignorados
function random_indexes(n, size, ignore=Int[])
    indexes = filter(x -> !(x in ignore), collect(1:size))
    @assert length(indexes) >= n "Not enough indexes to select from"
    my_shuffle!(indexes)
    return n == 1 ? indexes[1] : indexes[1:n]
end

function my_shuffle!(a::Vector{T}) where T
    n = length(a)
    for i in 1:n-1
        j = rand(i:n)
        a[i], a[j] = a[j], a[i]
    end
    return a
end

end
