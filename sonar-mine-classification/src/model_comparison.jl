using CSV
using DataFrames
using Random
using Flux
using Flux.Losses
using Statistics
using ScikitLearn: @sk_import, fit!, predict
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

#cd("C:/Users/niico/OneDrive/Documentos/GCED/2GCED/Q4/AA1/Prácticas/AA1-P2")


# Establecer la semilla para poder reproducirlo
Random.seed!(123)
# ————————————————————————————————
# 1) CARGA Y PREPARACIÓN
# ————————————————————————————————
include("Soluciones1.jl")
df = DataFrame(CSV.File("sonar.all-data", header=false))

# Separamos R y M para extraer características y etiquetas
df_rocks = df[df[:, end] .== "R", 1:end-1]
df_mines = df[df[:, end] .== "M", 1:end-1]

labels_rocks = repeat(["R"], nrow(df_rocks))
labels_mines = repeat(["M"], nrow(df_mines))

X = vcat(Matrix(df_rocks), Matrix(df_mines))
y = vcat(labels_rocks, labels_mines)



# Definimos una función para imprimir los resultados de cada prueba
function resultados(config, results)
    println("Resultados:")
    println("  Precisión (avg, std): ", results[1])
    println("  Error rate (avg, std): ", results[2])
    println("  Sensivity (avg, std): ", results[3])
    println("  Specificity (avg, std)", results[4])
    println("  VPP (avg, std)", results[5])
    println("  VPN (avg, std)", results[6])
    println("  Puntuación F1 (avg, std): ", results[7])
    println("  ConfusionMatrix,", results[8])
    println()
end

# ————————————————————————————————
# 2) Codificar y normalizar
# ————————————————————————————————
y_enc = oneHotEncoding(y)               # Bool matrix 208×2
#X   = normalizeMinMax(X)               # En principio no normalizamos ya que los valores están ya entre [0,1] 


# ————————————————————————————————
# 3) Índices de k‑fold CV
# ————————————————————————————————
k = 8
indices_cv = crossvalidation(y, k)   # tu función crossvalidation


# ————————————————————————————————
# 4) Experimentación con RR.NN.AA
# ————————————————————————————————

rna_topologies = [
    [8], [16], [32], [64],
    [8,2], [16,2], [32,2], [64,2]
]

for (i, topo) in enumerate(rna_topologies)
    println("=== Arquitectura ANN $i: ", topo, " ===")

    hyper_nn = Dict(
        "topology"     => topo,
        "maxEpochs"    => 100,
        "learningRate" => 0.01
    )

    resultados_nn = modelCrossValidation(
    :ANN,
    hyper_nn,
    (X, y),       
    indices_cv
    )
    # Imprimir los resultados usando la función print_results
    resultados(hyper_nn, resultados_nn)

end


# ————————————————————————————————
# 5) Experimentación con SVC 
# ————————————————————————————————

#Definimos 8 configuraciones diferentes de hiperparámetros de SVM
configuraciones_SVM = [
    Dict("C" => 0.01, "kernel" => "linear"),                                               # C muy bajo, margen grande
    Dict("C" => 1.0, "kernel" => "rbf", "gamma" => 0.01),                                  # gamma más bajo
    Dict("C" => 10.0, "kernel" => "poly", "degree" => 3, "gamma" => 0.1, "coef0" => 1.0),  # poly más complejo
    Dict("C" => 0.5, "kernel" => "linear"),                                                # valor intermedio de C
    Dict("C" => 1.0, "kernel" => "sigmoid", "gamma" => 0.1, "coef0" => 0.5),               # kernel menos común
    Dict("C" => 5.0, "kernel" => "rbf", "gamma" => 0.5),                                   # C más alto, gamma alto
    Dict("C" => 1.0, "kernel" => "poly", "degree" => 4, "gamma" => 0.05, "coef0" => 0.0),  # poly más profundo
    Dict("C" => 0.1, "kernel" => "rbf", "gamma" => 1.0),                                   # C bajo, gamma muy alto
]

#Pasamos a iterar sobre estas configuraciones llamando a la función modelCrossValidation
for (i, configuracion) in enumerate(configuraciones_SVM)
    
    resultados_SVM= modelCrossValidation(:SVC, configuracion,(X,y), indices_cv)
    
    println("=== Configuración SVM $i: ", configuracion, " ===")
    resultados(configuracion, resultados_SVM)

end



# ————————————————————————————————
# 6) Experimentación con DoME 
# ————————————————————————————————


#ESTA PARTE AUN NO FUNCIONA

# Definimos 8 valores para el número de nodos
valores_nodos = [5, 10, 15, 20, 25, 30, 35, 40]

for nodos in valores_nodos

    configuracion= Dict("maximumNodes"=> nodos, "random_state" => 1)   
    resultados_dome = modelCrossValidation(:DoME, configuracion, (X, y), indices_cv)
    

    println("Número de nodos: ", nodos)
    resultados(configuracion, resultados_dome)

end



# ————————————————————————————————
# 7) Experimentación con Árboles de Decisión 
# ————————————————————————————————

# Definimos 6 valores de profundidad
valores_profundidad= [2,4,6,8,10,12]

#Pasamos a iterar sobre estas profundidades llamando a la función modelCrossValidation
for valor in valores_profundidad

    configuracion = Dict("max_depth"=> valor, "random_state"=> 1)
    resultados_arbol= modelCrossValidation(:DecisionTreeClassifier, configuracion, (X,y), indices_cv)

    println("Profundidad: ", valor)
    resultados(configuracion, resultados_arbol)

end



# ————————————————————————————————
# 8) Experimentación con kNN 
# ————————————————————————————————

#Definimos 6 valores de k distintos

k_valor= [1,3,5,7,9,11]

#Pasamos a iterar sobre los valores de k llamando a la función modelCrossValidation
for k in k_valor

    configuracion= Dict("n_neighbors"=> k)
    resultados_kNN= modelCrossValidation(:KNeighborsClassifier, configuracion, (X, y), indices_cv)

    println("Valor de k: ", k)
    resultados(configuracion, resultados_kNN)

end

################################################################################################################
#                                                GRÁFICAS
################################################################################################################
using Plots

values = [nrow(df_rocks), nrow(df_mines)]
positions = [1, 1.2]

bar(
    positions, values;
    fillcolor = RGB(177/255, 0, 114/255),
    linecolor = :transparent,
    legend = false,
    ylabel = "Número de muestras",
    ylim = (0, maximum(values) + 20),
    xticks = (positions, ["Roca", "Mina"]),
    yguidefont = font(9, :gray),
    guidefontcolor = :gray,
    tickfontcolor = :gray,
    size = (400, 400),
    bar_width = 0.1
)

annotate!([(x, v + 4, text(string(v), 9, :gray)) for (x, v) in zip(positions, values)])

savefig("muestras.png")

using Statistics
using Plots

# Convertir a matriz numérica
matrix_rocks = Matrix(df_rocks)
matrix_mines = Matrix(df_mines)

# Calcular medias y desviaciones
mean_rock = vec(mean(matrix_rocks, dims=1))
std_rock = vec(std(matrix_rocks, dims=1))

mean_mine = vec(mean(matrix_mines, dims=1))
std_mine = vec(std(matrix_mines, dims=1))

# Gráfico
plot(
    1:60, mean_rock;
    ribbon=std_rock,
    fillalpha=0.2,
    label="Media Roca ± SD",
    color="#007bff",
    linewidth=2,
    legend=:topleft,
    xlabel="Frecuencia",
    ylabel="Amplitud",
    xguidefont=font(9, :gray),
    yguidefont=font(9, :gray),
    guidefontcolor=:gray,
    tickfontcolor=:gray,
    size=(800,400),
)

plot!(
    1:60, mean_mine;
    ribbon=std_mine,
    fillalpha=0.2,
    label="Media Mina ± SD",
    color="#ff7f0e",
    linewidth=2
)

