module LocalSearch

using Optim

export lbfgsb

# Define a wrapper function to use L-BFGS-B from Optim.jl
function lbfgsb(fun, x0, lower_bounds, upper_bounds, max_evals)
    result = optimize(
        fun,
        x0,
        LBFGS(),
        Optim.Options(store_trace = true, show_trace = true, maxf_evals = max_evals),
        lower = lower_bounds,
        upper = upper_bounds
    )
    minimizer = Optim.minimizer(result)
    num_evals = Optim.f_calls(result)
    return minimizer, num_evals
end

end