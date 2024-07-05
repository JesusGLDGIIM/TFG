using Optim

export lbfgsb

# Define a wrapper function to use L-BFGS-B from Optim.jl
function lbfgsb(fun, x0, lower_bounds, upper_bounds, max_evals)
    result = Optim.optimize(
        fun,
        lower_bounds,
        upper_bounds,
        x0,
        Optim.Fminbox(Optim.LBFGS()),
        Optim.Options(store_trace = true, show_trace = false, f_calls_limit = max_evals)
    )
    minimizer = Optim.minimizer(result)
    num_evals = Optim.f_calls(result)
    return minimizer, num_evals
end