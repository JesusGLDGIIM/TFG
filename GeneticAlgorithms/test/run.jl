# import Pkg
# Pkg.activate("..")  # Activa el entorno en el directorio actual

using Printf
using Random
# using Statistics
using MAT
# using Plots
using GeneticAlgorithms
using GeneticAlgorithms.DifferentialEvolution
using GeneticAlgorithms.MyLocalSearch
using GeneticAlgorithms.AbstractAlgorithm
using GeneticAlgorithms.Utils
using GeneticAlgorithms.Memetic
using GeneticAlgorithms.VariableGrouping

include("./cec2013lsgo.jl")

# Configuración por defecto
const NP = 100
evals = [Int(1.2e5), Int(6e5), Int(3e6)]
const maxevals = Int(3e6)
const H = 100

# Definir la función de límites
function get_bounds(func_num)
    if func_num in [1, 4, 7, 8, 11, 12, 13, 14, 15]
        return -100.0, 100.0
    elseif func_num in [2, 5, 9]
        return -5.0, 5.0
    elseif func_num in [3, 6, 10]
        return -32.0, 32.0
    else
        error("Función no reconocida")
    end
end

# Función para evaluar el algoritmo
function evaluate_shadeils(NP, runs, evals, func_num)
    # results = Dict{Int, Vector{Float64}}()
    lb, ub = get_bounds(func_num)

    D = 1000  # Dimensión del problema

    if(func_num == 13)
        D = 905
    end

    groups = [[i for i in 1:D]] 

    initial_eval = 0

    cec2013_init(func_num, "../../test/cec2013lsgo")
    cec2013_set_data_dir("../../test/cec2013lsgo/cdatafiles")

    for run in 1:runs
        funinfo = Dict("lower" => lb, "upper" => ub)
        
        fitness_fun = (f = x -> cec2013_eval_sol(x))

        result, shade = shadeils(fitness_fun, funinfo, D, evals, initial_eval, groups, NP)
        # if !haskey(results, func_num)
        #     results[func_num] = Float64[]
        # end
        # push!(results[func_num], result.fitness)
        # 
        # println("Run $run: Best fitness = $(result.fitness)")

        cec2013_next_run()
    end
    
    #for func_num in keys(results)
    #    mean_result = mean(results[func_num])
    #    std_result = std(results[func_num])
    #    println("Func $func_num - Mean fitness: $mean_result, Std fitness: $std_result")
    #end

    #

    #return results
end

function evaluate_shade(NP, runs, maxevals, H, func_num)
    # results = Dict{Int, Vector{Float64}}()
    lb, ub = get_bounds(func_num)
    D = 1000  # Dimensión del problema

    if(func_num == 13)
        D = 905
    end

    groups = [i for i in 1:D]

    initial_eval = 0

    cec2013_init(func_num, "../../test/cec2013lsgo")
    cec2013_set_data_dir("../../test/cec2013lsgo/cdatafiles")

    for run in 1:runs
        fitness_fun = x -> cec2013_eval_sol(x)
        bound = [lb, ub]
        
        shade = SHADE(bound, D, NP, H, maxevals, initial_eval)
        init(shade, fitness_fun, H)
        update(shade, fitness_fun, groups, maxevals)
        #if !haskey(results, func_num)
        #    results[func_num] = Float64[]
        #end
        #push!(results[func_num], best_fitness(shade))
        #
        #println("Run $run: Best fitness = $(best_fitness(shade))")

        cec2013_next_run()

        #plot(shade.num_eval_history, shade.best_fitness_history, xlabel="Evaluaciones", ylabel="Mejor Fitness", title="Convergencia de SHADE")
    end

    # for func_num in keys(results)
    #     mean_result = mean(results[func_num])
    #     std_result = std(results[func_num])
    #     println("Func $func_num - Mean fitness: $mean_result, Std fitness: $std_result")
    # end

    # return results
end

# Función para evaluar el algoritmo
function evaluate_ERDGshadeils(NP, runs, evals, func_num)
    lb, ub = get_bounds(func_num)

    D = 1000  # Dimensión del problema

    if(func_num == 13)
        D = 905
    end

    cec2013_init(func_num, "../../test/cec2013lsgo")
    cec2013_set_data_dir("../../test/cec2013lsgo/cdatafiles")

    funinfo = Dict("lower" => lb, "upper" => ub)
        
    fitness_fun = (f = x -> cec2013_eval_sol(x))

    lbv = -5.0 * ones(D)
    ubv = 5.0 * ones(D)

    groups, initial_eval = ERDG(fitness_fun, D, lbv, ubv)
    
    for run in 1:runs
        result, shade = shadeils(fitness_fun, funinfo, D, evals, initial_eval, groups, NP)

        cec2013_next_run()
    end
end

function evaluate_ERDGshade(NP, runs, maxevals, H, func_num)
    lb, ub = get_bounds(func_num)
    D = 1000  # Dimensión del problema

    if(func_num == 13)
        D = 905
    end

    cec2013_init(func_num, "../../test/cec2013lsgo")
    cec2013_set_data_dir("../../test/cec2013lsgo/cdatafiles")

    fitness_fun = (f = x -> cec2013_eval_sol(x))

    lbv = -5.0 * ones(D)
    ubv = 5.0 * ones(D)

    groups, initial_eval = ERDG(fitness_fun, D, lbv, ubv)

    for run in 1:runs
        bound = [lb, ub]
        shade = SHADE(bound, D, NP, H, maxevals, initial_eval)
        init(shade, fitness_fun, H)
        for group in groups
            factor = length(group)/D
            partialevals = (maxevals-initial_eval)*factor    
            update(shade, fitness_fun, group, partialevals)
        end    
        cec2013_next_run()
    end
end

function evaluate_DG2shadeils(NP, runs, evals, func_num)
    lb, ub = get_bounds(func_num)

    D = 1000  # Dimensión del problema

    if(func_num == 13)
        D = 905
    end

    cec2013_init(func_num, "../../test/cec2013lsgo")
    cec2013_set_data_dir("../../test/cec2013lsgo/cdatafiles")

    funinfo = Dict("lower" => lb, "upper" => ub)
        
    fitness_fun = (f = x -> cec2013_eval_sol(x))

    lbv = -5.0 * ones(D)
    ubv = 5.0 * ones(D)

    groups, initial_eval = DG2(fitness_fun, D, lb, ub)
    
    for run in 1:runs
        result, shade = shadeils(fitness_fun, funinfo, D, evals, initial_eval, groups, NP)

        cec2013_next_run()
    end
end

function evaluate_DG2shade(NP, runs, maxevals, H, func_num)
    lb, ub = get_bounds(func_num)
    D = 1000  # Dimensión del problema

    if(func_num == 13)
        D = 905
    end

    cec2013_init(func_num, "../../test/cec2013lsgo")
    cec2013_set_data_dir("../../test/cec2013lsgo/cdatafiles")

    fitness_fun = (f = x -> cec2013_eval_sol(x))

    lbv = -5.0 * ones(D)
    ubv = 5.0 * ones(D)

    groups, initial_eval = DG2(fitness_fun, D, lb, ub)

    for run in 1:runs
        bound = [lb, ub]
        shade = SHADE(bound, D, NP, H, maxevals, initial_eval)
        init(shade, fitness_fun, H)
        for group in groups
            factor = length(group)/D
            partialevals = (maxevals-initial_eval)*factor    
            update(shade, fitness_fun, group, partialevals)
        end    
        cec2013_next_run()
    end
end

function compare_grouping(func_num)
    # Definir límites y dimensiones según func_num
    lb, ub = get_bounds(func_num)
    D = 1000  # Dimensión del problema

    if func_num == 13
        D = 905
    end

    # Inicializar la función de prueba CEC2013
    cec2013_init(func_num, "../../test/cec2013lsgo")
    cec2013_set_data_dir("../../test/cec2013lsgo/cdatafiles")

    # Definir la función de fitness
    fitness_fun = x -> cec2013_eval_sol(x)

    # Crear límites de búsqueda en forma de vectores
    lbv = lb * ones(D)
    ubv = ub * ones(D)

    # Ejecutar ERDG y DG2
    erdg_groups, erdg_evals = ERDG(fitness_fun, D, lbv, ubv)
    dg2_groups, dg2_evals = DG2(fitness_fun, D, lb, ub)

    # Mostrar evaluaciones de función
    println("Función: ", func_num)
    println("Evaluaciones ERDG: ", erdg_evals)
    println("Evaluaciones DG2: ", dg2_evals)

    function normalize_groups(groups)
        return [sort(collect(Set(g))) for g in groups]
    end
    
    erdg_normalized = normalize_groups(erdg_groups)
    dg2_normalized = normalize_groups(dg2_groups)
    

    # Comparar los grupos obtenidos, independientemente del orden
    function compare_groups(groups1, groups2)
        sets1 = [Set(g) for g in groups1]
        sets2 = [Set(g) for g in groups2]
        
        for s1 in sets1
            if !(s1 in sets2)
                println("Grupo en ERDG no encontrado en DG2: ", s1)
            end
        end
        for s2 in sets2
            if !(s2 in sets1)
                println("Grupo en DG2 no encontrado en ERDG: ", s2)
            end
        end
        
        return all(s1 in sets2 for s1 in sets1) && all(s2 in sets1 for s2 in sets2)
    end

    

    if compare_groups(erdg_normalized, dg2_normalized)
        println("Son iguales")
    else
        println("Grupos ERDG: ", erdg_normalized)
        println("Grupos DG2: ", dg2_normalized)
    end
end


# Lee argumentos de la línea de comandos
function main()
    if length(ARGS) < 3
        println("Uso: julia script.jl <algorithm> <runs> <func_num>")
        return
    end

    # Argumentos
    algorithm = ARGS[1]  # Algoritmo: "shadeils" o "shade"
    runs = parse(Int, ARGS[2])
    func_num = parse(Int, ARGS[3])

    # Ejecuta el algoritmo seleccionado
    if algorithm == "shadeils"
        results = evaluate_shadeils(NP, runs, evals, func_num)
    elseif algorithm == "shade"
        results = evaluate_shade(NP, runs, maxevals, H, func_num)
    elseif algorithm == "ERDGshade"
        results = evaluate_ERDGshade(NP, runs, maxevals, H, func_num)
    elseif algorithm == "ERDGshadeils"
        results = evaluate_ERDGshadeils(NP, runs, evals, func_num)
    elseif algorithm == "DG2shade"
        results = evaluate_ERDGshade(NP, runs, maxevals, H, func_num)
    elseif algorithm == "DG2shadeils"
        results = evaluate_ERDGshadeils(NP, runs, evals, func_num)
    elseif algorithm == "Grouping"
        compare_grouping(func_num)
    else
        println("Algoritmo no reconocido. Use 'shadeils', 'shade', 'ERDGshade' o 'ERDGshadeils'.")
    end
end

# Ejecuta la función principal
main()
