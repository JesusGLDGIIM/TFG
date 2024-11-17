export DG2

# Implementación del algoritmo DG2 en Julia
function DG2(fun, dim, lb, ub)
    groups = []  # Lista para almacenar los grupos de variables
    ungrouped_vars = collect(1:dim)  # Variables aún no agrupadas
    FEs = 0  # Contador de evaluaciones de función

    # Punto central en el espacio de búsqueda
    x0 = (ub .+ lb) ./ 2
    f0 = fun(x0)
    FEs += 1

    # Proceso de agrupamiento diferencial
    while !isempty(ungrouped_vars)
        xi = ungrouped_vars[1]
        current_group = [xi]
        ungrouped_vars = setdiff(ungrouped_vars, xi)

        xi_vars = copy(ungrouped_vars)
        vars_to_remove = []

        # Verificar interacción con cada variable restante
        for xj in xi_vars
            is_interacting, FEs = interaction_test(fun, xi, xj, x0, f0, lb, ub, FEs)
            if is_interacting
                push!(current_group, xj)
                push!(vars_to_remove, xj)
            end
        end

        # Remover variables ya agrupadas y agregar el grupo a la lista final
        ungrouped_vars = setdiff(ungrouped_vars, vars_to_remove)
        push!(groups, current_group)
    end

    return groups, FEs
end

# Función para probar la interacción entre dos variables
function interaction_test(fun, xi, xj, x0, f0, lb, ub, FEs)
    # Parámetros de interacción y error de máquina
    muM = eps() / 2
    n = length(x0)
    gamma_n = (n * muM) / (1 - n * muM)

    # Pequeña perturbación para evaluar interacciones
    delta = 0.01 * (ub - lb)

    # Crear perturbaciones en x_i y x_j
    x_i = copy(x0)
    x_i[xi] += delta[xi]

    x_j = copy(x0)
    x_j[xj] += delta[xj]

    x_ij = copy(x0)
    x_ij[xi] += delta[xi]
    x_ij[xj] += delta[xj]

    # Evaluar función en los puntos perturbados
    f_i = fun(x_i)
    f_j = fun(x_j)
    f_ij = fun(x_ij)
    FEs += 3

    # Calcular las diferencias
    delta_i = f_i - f0
    delta_j = f_j - f0
    delta_ij = f_ij - f0

    # Umbral de interacción
    Fmax = abs(delta_i) + abs(delta_j) + abs(delta_ij)
    epsilon = gamma_n * Fmax

    # Verificar si hay interacción
    if abs(delta_i + delta_j - delta_ij) > epsilon
        return true, FEs  # Las variables interactúan
    else
        return false, FEs  # Las variables son separables
    end
end


#=
# Función objetivo de prueba
function test_fun(x)
    # Función de ejemplo con interacciones entre variables
    return sum(x .^ 2) + (25 - x[1]*x[2])^2 + (x[1] + x[2])^2 + (25 - x[3]*x[4])^2 + (x[3] + x[4])^2
end

# Probando el algoritmo DG2
using .VariableGrouping

dim = 100
lb = -5.0 * ones(dim)
ub = 5.0 * ones(dim)

groups, fEvalNum = DG2(test_fun, dim, lb, ub)

println("Grupos: ", groups)
println("Evaluaciones de la función: ", fEvalNum)
=#
