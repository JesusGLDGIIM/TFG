using Metaheuristics

export mts_ls_1

# Define a wrapper function to use MTS-LS-1 from Metaheuristics.jl
function mts_ls_1(fun, x0, lower_bounds, upper_bounds, max_evals)
    result = Metaheuristics.optimize(
        Metaheuristics.MTS_LS1(max_evaluations = max_evals),
        fun,
        lower_bounds,
        upper_bounds,
        x0
    )
    minimizer = result.minimizer
    num_evals = result.num_evaluations
    return minimizer, num_evals
end