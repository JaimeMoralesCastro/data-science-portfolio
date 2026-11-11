begin
    using CSV
    using Plots
    using DataFrames
    using Statistics
    using Flux
    using Flux.Losses
    using StatsPlots
    using Random
    using PrettyTables
    using Serialization
end


include("Soluciones_Practica2.jl")

# ============================
# CARGA Y PREPROCESAMIENTO
# ============================

# Leer datos
dataset = CSV.File("wine.csv")
df = DataFrame(dataset)


# Estadísticas básicas
print(describe(df))

display(histogram(df.Alcohol, bins=20,normalize = true,xlabel="Alcohol", ylabel="Frecuencia", title="Histograma de Alcohol"))
display(density!(df.Alcohol))

display(histogram(df.Malic_acid, bins=20,normalize = true, xlabel="Malic Acid", ylabel="Frecuencia", title="Histograma de Ácido Málico"))
display(density!(df.Malic_acid))

display(histogram(df.ash, bins=20,normalize = true, xlabel="Ash", ylabel="Frecuencia", title="Histograma de residuos"))
display(density!(df.ash))

display(histogram(df.alcalinity_of_ash, bins=20,normalize = true, xlabel="Alcalinidad de los residuos", ylabel="Frecuencia", title="Histograma de Alcalinidad de Cenizas"))
display(density!(df.alcalinity_of_ash))

display(histogram(df.Magnesium, bins=20,normalize = true, xlabel="Magnesio", ylabel="Frecuencia", title="Histograma de Magnesio"))
display(density!(df.Magnesium))

display(histogram(df.Total_phenols, bins=20,normalize = true, xlabel="Fenoles Totales", ylabel="Frecuencia", title="Histograma de Fenoles Totales"))
display(density!(df.Total_phenols))

display(histogram(df.Flavanoids, bins=20,normalize = true, xlabel="Flavonoides", ylabel="Frecuencia", title="Histograma de Flavonoides"))
display(density!(df.Flavanoids))

display(histogram(df.Nonflavanoid_phenols,normalize = true, bins=20, xlabel="Fenoles No Flavonoides", ylabel="Frecuencia", title="Histograma de Fenoles No Flavonoides"))
display(density!(df.Nonflavanoid_phenols))

display(histogram(df.Proantho, bins=20,normalize = true, xlabel="Proantocianidinas", ylabel="Frecuencia", title="Histograma de Proantocianidinas"))
display(density!(df.Proantho))

display(histogram(df.Color_intensity,normalize = true, bins=20, xlabel="Intensidad de Color", ylabel="Frecuencia", title="Histograma de Intensidad de Color"))
display(density!(df.Color_intensity))
 
display(histogram(df.HUE, bins=20,normalize = true, xlabel="Tono (Hue)", ylabel="Frecuencia", title="Histograma de Tono (Hue)"))
display(density!(df.HUE))

display(histogram(df.OD280Od315,bins=20,normalize = true, xlabel="OD280/OD315", ylabel="Frecuencia", title="Histograma de OD280/OD315"))
display(density!(df.OD280Od315))

display(histogram(df.Proline, bins=20,normalize = true, xlabel="Prolina", ylabel="Frecuencia", title= "Histograma de Prolina"))
display(density!(df.Proline))
 

# Boxplot de ejemplo de outlier
display(boxplot(df.Group, df.Magnesium, xlabel="Group", ylabel="Magnesium", title = "Boxplot del magnesio"))



# Datos sin el target
data = Matrix(select(df, Not(:Group)))

print("=====Matriz de correlaciones=====")
# Matriz de correlación
print(cor(Matrix(df)))



# Normalización Min-Max
X_minmax = normalizeMinMax(data)


# Normalización Zero-Mean
X_zeroMean = normalizeZeroMean(data)
println("\n=== Normalización Zero-Mean (media 0, varianza 1) ===")
println(X_zeroMean)

# Definir entradas y salidas
inputs = X_zeroMean
targets = df[:, :Group]

# Validación cruzada

Random.seed!(1234)  # Puedes poner cualquier número entero

# num_folds = 5
# cv_results = crossvalidation(targets, num_folds)

# Cargar los índices para repetir los experimentos exactamente
cv_results = deserialize("cv_indices.jls")

# ============================
# REDES NEURONALES ARTIFICIALES (ANN)
# ============================

architectures = [[5], [10], [15], [20], [10, 5], [15, 10], [20, 5], [25, 10]]
results_ann = []

for arch in architectures
    params = Dict{String, Any}(
        "topology" => arch,
        "learningRate" => 0.01,
        "validationRatio" => 0.2,
        "numExecutions" => 5,
        "maxEpochs" => 150,
        "maxEpochsVal" => 15
    )
    result = modelCrossValidation(:ANN, params, (inputs, targets), cv_results)
    push!(results_ann, (arch, result))
end


# Métricas ANN


f1_scores = [result[2][7][1] for result in results_ann]
desv_f1_scores = [result[2][7][2] for result in results_ann]



Accuracy_ann = [result[2][1][1] for result in results_ann]
desv_Accuracy_ann = [result[2][1][2] for result in results_ann]



scatter(1:length(Accuracy_ann), Accuracy_ann, yerr=desv_Accuracy_ann, xlabel="arquitecturas", ylabel="Accuracy", xticks=(1:length(architectures), architectures),
    markershape=:circle, markercolor=:blue, legend=false, 
    title="Accuracy para Ann" )

scatter(1:length(f1_scores), f1_scores, yerr=desv_f1_scores, xlabel="arquitecturas", ylabel="f1-score", xticks=(1:length(architectures), architectures),
    markershape=:circle, markercolor=:blue, legend=false, 
    title="f1-score para Ann" )
# ============================
# SVM
# ============================

svm_hyperparams = [
    Dict("kernel" => "linear",  "C" => 0.1),
    Dict("kernel" => "linear",  "C" => 1.0),
    Dict("kernel" => "rbf",     "C" => 0.1, "gamma" => 0.1),
    Dict("kernel" => "rbf",     "C" => 1.0, "gamma" => 0.5),
    Dict("kernel" => "poly",    "C" => 0.1, "degree" => 2, "gamma" => 0.1, "coef0" => 0.0),
    Dict("kernel" => "poly",    "C" => 1.0, "degree" => 3, "gamma" => 0.5, "coef0" => 0.0),
    Dict("kernel" => "sigmoid", "C" => 0.1, "gamma" => 0.1, "coef0" => 0.0),
    Dict("kernel" => "sigmoid", "C" => 1.0, "gamma" => 0.5, "coef0" => 0.0)
]

results_svm = []

for params in svm_hyperparams 
    result = modelCrossValidation(:SVC, params, (inputs, targets), cv_results)
    push!(results_svm, (params, result))
end

f1_scores_svm = [result[2][7][1] for result in results_svm]
desv_f1_scores_svm = [result[2][7][2] for result in results_svm]

Accuracy_svm = [result[2][1][1] for result in results_svm]
desv_Accuracy_svm = [result[2][1][2] for result in results_svm]

display(scatter(1:length(Accuracy_svm), Accuracy_svm, yerr=desv_Accuracy_svm, ylabel="Accuracy",
    markershape=:circle, markercolor=:blue, legend=false, 
    title="Accuracy para SVM" ))

display(scatter(1:length(f1_scores_svm), f1_scores_svm, yerr=desv_f1_scores_svm, ylabel="F1score",
    markershape=:circle, markercolor=:blue, legend=false, 
    title="F1score para SVM" ))

# ============================
# ÁRBOLES DE DECISIÓN
# ============================

tree_depths = [2, 4, 6, 8, 10, 12]
results_tree = []

for depth in tree_depths
    params = Dict("max_depth" => depth)
    result = modelCrossValidation(:DecisionTreeClassifier, params, (inputs, targets), cv_results)
    push!(results_tree, (depth, result))
end


f1_scores_tree = [result[2][7][1] for result in results_tree]
desv_f1_scores_tree = [result[2][7][2] for result in results_tree]



Accuracy_tree = [result[2][1][1] for result in results_tree]
desv_Accuracy_tree = [result[2][1][2] for result in results_tree]


display(scatter(1:length(Accuracy_tree), Accuracy_tree, yerr=desv_Accuracy_tree, xlabel="Valores de Profundidad", ylabel="Accuracy", xticks=(1:length(tree_depths), tree_depths),
    markershape=:circle, markercolor=:blue, legend=false, 
    title="Accuracy para Árboles" ))

display(scatter(1:length(f1_scores_tree), f1_scores_tree, yerr=desv_f1_scores_tree, xlabel="Valores de Profundidad", ylabel="F1score", xticks=(1:length(tree_depths), tree_depths),
    markershape=:circle, markercolor=:blue, legend=false, 
    title="F1score para Árboles" ))
# ============================
# kNN
# ============================

knn_values = [1, 3, 5, 7, 9, 11, 14, 20, 30]
results_knn = []

for k in knn_values
    params = Dict("n_neighbors" => k)
    result = modelCrossValidation(:KNeighborsClassifier, params, (inputs, targets), cv_results)
    push!(results_knn, (k, result))
end

#Métricas KNN


f1_scores_knn = [result[2][7][1] for result in results_knn]
desv_f1_scores_knn = [result[2][7][2] for result in results_knn]


Accuracy_knn = [result[2][1][1] for result in results_knn]
desv_Accuracy_knn = [result[2][1][2] for result in results_knn]



display(scatter(1:length(Accuracy_knn), Accuracy_knn, yerr=desv_Accuracy_knn, xlabel="Valores de K", ylabel="Accuracy", xticks=(1:length(knn_values), knn_values),
    markershape=:circle, markercolor=:blue, legend=false, 
    title="Accuracy para KNN" ))

display(scatter(1:length(f1_scores_knn), f1_scores_knn, yerr=desv_f1_scores_knn, xlabel="Valores de K", ylabel="F1score", xticks=(1:length(knn_values), knn_values),
    markershape=:circle, markercolor=:blue, legend=false, 
    title="F1score para KNN" ))


# ============================
# DoME
# ============================

# Puedes ajustar estos valores según el intervalo óptimo que hayas observado
nodes_values = [20,40, 50, 60, 70, 80, 90, 100, 110]
results_dome = []

for n_nodes in nodes_values
    params = Dict("maximumNodes" => n_nodes)
    result = modelCrossValidation(:DoME, params, (inputs, targets), cv_results)
    push!(results_dome, (n_nodes, result))
end

# Métricas DoME

f1_scores_dome = [result[2][7][1] for result in results_dome]
desv_f1_scores_dome = [result[2][7][2] for result in results_dome]

Accuracy_dome = [result[2][1][1] for result in results_dome]
desv_Accuracy_dome = [result[2][1][2] for result in results_dome]



display(scatter(1:length(Accuracy_dome), Accuracy_dome, yerr=desv_Accuracy_dome, xlabel="Nodos", ylabel="Accuracy", xticks=(1:length(nodes_values), nodes_values),
    markershape=:circle, markercolor=:blue, legend=false, 
    title="Accuracy para DoME" ))

display(scatter(1:length(f1_scores_dome), f1_scores_dome, yerr=desv_f1_scores_dome, xlabel="Nodos", ylabel="F1score", xticks=(1:length(nodes_values), nodes_values),
    markershape=:circle, markercolor=:blue, legend=false, 
    title="F1score para DoME" ))


# ============================
# TABLAS RESUMEN FINALES
# ============================

# Crear DataFrames con métricas

df_ann = DataFrame(
    Arquitectura=string.(architectures),
    Accuracy=Accuracy_ann,
    Desv_Accuracy=desv_Accuracy_ann,
    F1_Score=f1_scores,
    Desv_F1=desv_f1_scores
)

df_svm = DataFrame(
    Kernel=[params["kernel"] for (params, _) in results_svm],
    Parametros=string.([params for (params, _) in results_svm]),
    Accuracy=Accuracy_svm,
    Desv_Accuracy=desv_Accuracy_svm,
    F1_Score=f1_scores_svm,
    Desv_F1=desv_f1_scores_svm
)

df_tree = DataFrame(
    Profundidad=tree_depths,
    Accuracy=Accuracy_tree,
    Desv_Accuracy=desv_Accuracy_tree,
    F1_Score=f1_scores_tree,
    Desv_F1=desv_f1_scores_tree
)

df_knn = DataFrame(
    K=knn_values,
    Accuracy=Accuracy_knn,
    Desv_Accuracy=desv_Accuracy_knn,
    F1_Score=f1_scores_knn,
    Desv_F1=desv_f1_scores_knn
)

df_dome = DataFrame(
    Nodos=nodes_values,
    Accuracy=Accuracy_dome,
    Desv_Accuracy=desv_Accuracy_dome,
    F1_Score=f1_scores_dome,
    Desv_F1=desv_f1_scores_dome
)

# Mostrar tablas

println("\n=== Resultados ANN ===")
pretty_table(df_ann)

println("\n=== Resultados SVM ===")
pretty_table(df_svm)

println("\n=== Resultados Árboles ===")
pretty_table(df_tree)

println("\n=== Resultados kNN ===")
pretty_table(df_knn)

println("\n=== Resultados DoME ===")
pretty_table(df_dome)
