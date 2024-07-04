module GeneticAlgorithms

    include("AbstractAlgorithm.jl")
    include("Utils.jl")

    # Local Search Algorithms
    include("LocalSearch/mts_ls_1.jl")
    include("LocalSearch/lbfgsb.jl")

    # Differential Evolution Algorithms
    include("DifferentialEvolution/shade.jl")

    # Variable Grouping Algorithms
    include("VariableGrouping/dg2.jl")
    include("VariableGrouping/erdg.jl")

    # Hybrid Algorithms
    include("Memetic/dg2_shade_ils.jl")
    include("Memetic/erdg_shade_ils.jl")
    include("Memetic/shade_ils.jl")

    using .AbstractAlgorithm
    using .Utils
    # using .LocalSearch
    using .DifferentialEvolution
    # using .VariableGrouping
    # using .Memetic

    export VariableGroupingAlgorithm, 
           GeneralAlgorithm, 
           get_groups, 
           num_groups, 
           group_sizes,
           population_size, 
           num_dimensions, 
           best_fitness, 
           best_solution, 
           all_fitness, 
           all_population, 
           bounds, 
           init, 
           update,
           random_population, 
           clip, 
           EAresult, 
           get_experiments_file, 
           random_indexes

end