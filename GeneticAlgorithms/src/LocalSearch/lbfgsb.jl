using Optim

export lbfgsb

# Define a wrapper function to use L-BFGS-B from Optim.jl
function lbfgsb(fun, x0, lower_bounds, upper_bounds, max_evals, state)
    result = Optim.optimize(
        x -> fun(x, state),
        lower_bounds,
        upper_bounds,
        x0,
        Optim.Fminbox(Optim.LBFGS()),
        Optim.Options(store_trace = true, show_trace = true, f_calls_limit = max_evals)
    )
    minimizer = Optim.minimizer(result)
    fitness = Optim.minimum(result)
    num_evals = Optim.f_calls(result)
    return minimizer, fitness, num_evals
end