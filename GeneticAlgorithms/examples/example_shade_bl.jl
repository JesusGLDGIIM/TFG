using GeneticAlgorithms
using GeneticAlgorithms.AbstractAlgorithm
using GeneticAlgorithms.DifferentialEvolution
using GeneticAlgorithms.MyLocalSearch

# Función de ejemplo para SHADE
function example_SHADE_with_local_search()
    # Definir la función de objetivo
    function objective_function(x::Vector{Float64})
        return sum(x .^ 2)
    end

    # Parámetros de SHADE
    bounds = [-5.0, 5.0]
    dim = 10
    pop_size = 50
    H = 10
    max_evals = 8000
    local_search_max_evals = 1000  # Límite de evaluaciones para la búsqueda local
    SR = fill(0.5, dim)

    # Crear instancia de SHADE
    algo = DifferentialEvolution.SHADE(bounds, dim, pop_size, H, max_evals)

    # Inicializar SHADE
    AbstractAlgorithm.init(algo, objective_function)

    # Ejecutar SHADE
    algo = AbstractAlgorithm.update(algo, objective_function)

    # Obtener la mejor solución de SHADE
    best_sol_index = argmin(algo.fitness)
    best_sol = algo.population[:, best_sol_index]

    # Verificar límites
    lower_bounds = fill(bounds[1], dim)
    upper_bounds = fill(bounds[2], dim)

    # Aplicar L-BFGS-B
    local_search_sol_lbfgsb, lbfgsb_evals = MyLocalSearch.lbfgsb(objective_function, best_sol, lower_bounds, upper_bounds, local_search_max_evals)

    # Aplicar MTS-LS-1
    local_search_sol_mts_ls_1, mts_ls_1_evals = MyLocalSearch.mtsls(objective_function, best_sol, AbstractAlgorithm.best_fitness(algo), lower_bounds[1], upper_bounds[1], local_search_max_evals, SR)

    # Elegir la mejor solución después de la búsqueda local
    if objective_function(local_search_sol_lbfgsb) < objective_function(local_search_sol_mts_ls_1.solution)
        final_best_sol = local_search_sol_lbfgsb
        final_evals = lbfgsb_evals
    else
        final_best_sol = local_search_sol_mts_ls_1
        final_evals = mts_ls_1_evals
    end

    # Actualizar la población original con la mejor solución encontrada por la búsqueda local
    algo.population[:, best_sol_index] = final_best_sol
    algo.fitness[best_sol_index] = objective_function(final_best_sol)
    algo.best_sol = final_best_sol
    algo.best_fit = objective_function(final_best_sol)

    # Imprimir resultados
    println("Best solution from SHADE: ", best_sol)
    println("Best fitness from SHADE: ", AbstractAlgorithm.best_fitness(algo))
    println("Best solution after L-BFGS-B: ", local_search_sol_lbfgsb, " (Evaluations: ", lbfgsb_evals, ")")
    println("Best solution after MTS-LS-1: ", local_search_sol_mts_ls_1, " (Evaluations: ", mts_ls_1_evals, ")")
    println("Final best solution after local search: ", final_best_sol)
    println("Final best fitness: ", algo.best_fit)
    println("Total evaluations used by the local search: ", final_evals)
end

# Llamar a la función de ejemplo
example_SHADE_with_local_search()