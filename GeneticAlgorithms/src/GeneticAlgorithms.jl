module GeneticAlgorithms

# Local Search Algorithms
include("LocalSearch/mts_ls_1.jl")
include("LocalSearch/lbfgsb.jl")

# Differential Evolution Algorithms
include("DifferentialEvolution/shade.jl")

# Variable Grouping Algorithms
include("variablegrouping/dg2.jl")
include("variablegrouping/erdg.jl")

# Hybrid Algorithms
include("Memetic/dg2_shade_ils.jl")
include("Memetic/erdg_shade_ils.jl")
include("Memetic/shade_ils.jl")

# Export modules or functions
using .LocalSearch
using .DifferentialEvolution
using .VariableGrouping
using .Memetic

end