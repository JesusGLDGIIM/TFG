# ERDG algorithm implementation in Julia
function ERDG(fun, fun_number, dim, lb, ub)
    seps = []
    nongroups = []
    FEs = 0
    
    p1 = lb .* ones(dim)  # Element-wise multiplication
    y1 = fun(p1, fun_number)
    FEs += 1
    
    sub1 = [1]
    sub2 = collect(2:dim)
    
    while !isempty(sub2)

        p2 = copy(p1)
        for i in eachindex(sub1)
            p2[sub1[i]] = ub[sub1[i]]
        end
        y2 = fun(p2, fun_number)
        FEs += 1
        
        y = [y1, -y2, NaN, NaN]
        sub1_a, FEs, _ = INTERACT(fun, fun_number, sub1, sub2, p1, p2, FEs, ub, lb, y)
        
        if length(sub1_a) == length(sub1)
            if length(sub1) == 1
                push!(seps, sub1)
            else
                push!(nongroups, sub1)
            end
            sub1 = [sub2[1]]
            sub2 = sub2[2:end]
        else
            sub1 = sub1_a
            sub2 = setdiff(sub2, sub1)
        end
        
        if isempty(sub2)
            if length(sub1) <= 1
                push!(seps, sub1)
            else
                push!(nongroups, sub1)
            end
        end
    end
    
    nonseps = nongroups
    groups001 = Vector{Any}(undef, length(nonseps) + length(seps))
    for sepsIndex in eachindex(nonseps)
        groups001[sepsIndex] = nonseps[sepsIndex]
    end
    for sepsIndex in eachindex(seps)
        groups001[length(nonseps) + sepsIndex] = seps[sepsIndex]
    end
    fEvalNum = FEs
    return groups001, fEvalNum
end

function INTERACT(fun, fun_number, sub1, sub2, p1, p2, FEs, ub, lb, y)
    muM = eps() / 2
    gamma(n) = (n * muM) / (1 - n * muM)
    
    nonsepFlag = 1
    y001 = copy(y)
    if any(isnan.(y))
        p3 = copy(p1)
        p4 = copy(p2)
        for i in eachindex(sub2)
            p3[sub2[i]] = (ub[sub2[i]] + lb[sub2[i]]) / 2
            p4[sub2[i]] = (ub[sub2[i]] + lb[sub2[i]]) / 2
        end
        y3 = fun(p3, fun_number)
        y4 = fun(p4, fun_number)
        FEs += 2
        
        # Asegurarse de que y001 tenga suficiente longitud
        # if length(y001) < 4
        #     resize!(y001, 4)
        # end

        y001[3:4] = [-y3, y4]
        Fmax = sum(abs.(y001))
        epsilon = gamma(sqrt(length(ub)) + 2) * Fmax
        deltaDiff001 = abs(sum(y001))
        if deltaDiff001 <= epsilon
            nonsepFlag = 0
        end
    end
    
    if nonsepFlag == 1
        if length(sub2) == 1
            sub1 = union(sub1, sub2)
        else
            k = floor(Int, length(sub2) / 2)
            sub2_1 = sub2[1:k]
            sub2_2 = sub2[k+1:end]
            
            sub1_1, FEs, y002 = INTERACT(fun, fun_number, sub1, sub2_1, p1, p2, FEs, ub, lb, [y[1], y[2], NaN, NaN])
            deltaDiffDiff = sum(y001) - sum(y002)
            if deltaDiffDiff != 0
                if length(sub1_1) == length(sub1)
                    sub1_2, FEs, _ = INTERACT(fun, fun_number, sub1, sub2_2, p1, p2, FEs, ub, lb, y001)
                else
                    sub1_2, FEs, _ = INTERACT(fun, fun_number, sub1, sub2_2, p1, p2, FEs, ub, lb, [y[1], y[2], NaN, NaN])
                end
                sub1 = union(sub1_1, sub1_2)
            else
                sub1 = sub1_1
            end
        end
    end
    return sub1, FEs, y001
end

# Función objetivo de prueba
function test_fun(x, fun_number)
    # Ejemplo de una función simple: suma de cuadrados
    return sum(x .^ 2) + (25 - x[1]*x[2])^2 + (x[1] + x[2])^2
end

# Probando el algoritmo ERDG
dim = 10
lb = -5.0 * ones(dim)
ub = 5.0 * ones(dim)
fun_number = 1

groups, fEvalNum = ERDG(test_fun, fun_number, dim, lb, ub)

println("Groups: ", groups)
println("Function Evaluations: ", fEvalNum)
