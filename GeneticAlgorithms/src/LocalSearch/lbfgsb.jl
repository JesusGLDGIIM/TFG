using Optim

export lbfgsb

# Define a wrapper function to use L-BFGS-B from Optim.jl
function lbfgsb(fun, x0, lower_bounds, upper_bounds, max_evals)
    println("Initial x0: ", x0)
    println(lower_bounds)
    println(upper_bounds)
    result = Optim.optimize(
        fun,
        x0,
        # upper_bounds,
        # lower_bounds,
        # Optim.Fminbox(Optim.LBFGS()),
        Optim.Options(store_trace = true, show_trace = true, f_calls_limit = max_evals)
    )
    minimizer = Optim.minimizer(result)
    num_evals = Optim.f_calls(result)
    return minimizer, num_evals
end