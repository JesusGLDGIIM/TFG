using Libdl

lib::Ptr{Nothing} = Ptr{Nothing}()
dim::Int = 0

"""
    cec2013_init(funcid::Int, dir::String="test/cec2013lsgo")

Init the cec2013 benchmark for the funcid passed as parameter.

# Parameters:

- funcid: The id of the function to test.
- dir: Directory of the library.

"""
function cec2013_init(funcid::Int, dir::String="test/cec2013lsgo")
    @assert funcid >= 1 && funcid <= 15 "Error, funcid $(funcid) is not between 1 and 15"
    global lib
    lib_path = joinpath(dir, "libcec2013lsgo.dll")
    println("Loading library from $lib_path")
    lib = Libdl.dlopen(lib_path)
    if lib == C_NULL
        error("Failed to load library from $lib_path")
    end
    
    set_func = dlsym(lib, :set_func)
    if set_func == C_NULL
        error("Failed to load symbol 'set_func'")
    end
    
    ccall(set_func, Cvoid, (Cint,), funcid)
end

"""
    cec2013_set_data_dir(new_data_dir::String)

Set the data directory for cec2013 benchmark.

# Parameters:

- new_data_dir: Path to the new data directory.

"""
function cec2013_set_data_dir(new_data_dir::String)
    global lib
    @assert lib !== Ptr{Nothing}() "cec2013_init must be applied"
    set_data_dir = dlsym(lib, :set_data_dir)
    if set_data_dir == C_NULL
        error("Failed to load symbol 'set_data_dir'")
    end
    ccall(set_data_dir, Cvoid, (Cstring,), new_data_dir)
end

"""
    cec2013_eval_sol(sol::Vector{Float64}) -> Float64

Evaluate the solution vector.

# Parameters:

- sol: Vector representing the solution.

# Returns:

- The fitness value of the solution.

"""
function cec2013_eval_sol(sol::Vector{Float64})::Float64
    # global aux += 1
    # if(aux % 25000 == 0)
    #     println("Evaluaciones: ", aux)
    # end
    global lib
    @assert lib !== Ptr{Nothing}() "cec2013_init must be applied"
    eval_sol = dlsym(lib, :eval_sol)
    if eval_sol == C_NULL
        error("Failed to load symbol 'eval_sol'")
    end
    return ccall(eval_sol, Cdouble, (Ptr{Cdouble},), sol)
end

"""
    cec2013_free_func()

Free the resources allocated by the benchmark functions.

"""
function cec2013_free_func()
    global lib
    @assert lib !== Ptr{Nothing}() "cec2013_init must be applied"
    free_func = dlsym(lib, :free_func)
    if free_func == C_NULL
        error("Failed to load symbol 'free_func'")
    end
    ccall(free_func, Cvoid, ())
end

"""
    cec2013_next_run()

Proceed to the next run of the benchmark.

"""
function cec2013_next_run()
    global lib
    @assert lib !== Ptr{Nothing}() "cec2013_init must be applied"
    next_run = dlsym(lib, :next_run)
    if next_run == C_NULL
        error("Failed to load symbol 'next_run'")
    end
    ccall(next_run, Cvoid, ())
end


"""
    Used to optimize by parts
    For example, you have:
    - The vector [0, 8.5 3, 4, 9]
    - The function (x .^ 2) + (25 - x[1]*x[2])^2 + (x[1] + x[2])^2 + (25 - x[3]*x[4])^2 + (x[3] + x[4])^2
    - The function can be optimized optimizing [[x1,x2],[x3,x4],[x5]]
    - You can do f = wrapper_fun(f, dict_posi, dict_values)
    - x = [0,0] (nuevos valores para [x3, x4])
    - f = wrapper_fun(f, Dict(1=>3, 2=>4), Dict(1=>0, 2=>8.5, 5=>9))
    - f(x) will return f([0, 8.5, 0, 0, 9])
"""
function wrapper_fun(f, dict_posi, dict_values)
    function fit(sol)
        c = zeros(length(keys(dict_posi)) + length(keys(dict_values)))
        @show c
        for (i, new_posi) in dict_posi
            c[new_posi] = sol[i]
        end
        for (i, val) in dict_values
            c[i] = val
        end
        @show c
        res = f(c)
        return res
    end
end