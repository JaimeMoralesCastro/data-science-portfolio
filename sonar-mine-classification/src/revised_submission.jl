#############################################################
#  Práctica 2 – Segunda Oportunidad                         
#############################################################

#cd("C:/Users/jaimo/OneDrive/Documentos/Escritorio/P2-AA1/codigo")
using CSV           
using DataFrames     
using Random       
using Statistics     
using Plots          


Random.seed!(123)
include("Soluciones1.jl")   


# CARGA DE DATOS        


df = DataFrame(CSV.File("sonar.all-data", header=false))

df_rocks = df[df[:, end] .== "R", 1:end-1]
df_mines = df[df[:, end] .== "M", 1:end-1]

labels_rocks = repeat(["R"], nrow(df_rocks))
labels_mines = repeat(["M"], nrow(df_mines))

X = vcat(Matrix(df_rocks), Matrix(df_mines))
y = vcat(labels_rocks, labels_mines)

# CV        

y_enc = oneHotEncoding(y)    
k = 8                        
indices_cv = crossvalidation(y, k)

# FICHERO DE ÍNDICES
function save_all_indices(indices, archivo)
     open(archivo, "w") do file
         for fold in indices
             for idx in fold
                 println(file, idx)
             end
         end
     end
end

save_all_indices(indices_cv, "cv_indices.dat")


# IMPRIMIR RESULTADOS              

affix = " (avg, std): "
function show_results(cfg, res)
    println("Config: ", cfg)
    println("  Acc"*affix, res[1]) ; println("  Err"*affix, res[2])
    println("  Rec"*affix, res[3]) ; println("  Spec"*affix, res[4])
    println("  Prec"*affix, res[5]); println("  NPV"*affix, res[6])
    println("  F1 "*affix, res[7]) ; println("  Confusion Matrix:\n", res[8]) ; println()
end

# EXPERIMENTACIÓN    

# RNA – 3 arquitecturas
for topo in [[16], [32], [16,2]]
    cfg = Dict("topology"=>topo, "maxEpochs"=>100, "learningRate"=>0.01)
    res = modelCrossValidation(:ANN, cfg, (X, y), indices_cv)
    show_results(cfg, res)
end

# SVC – 3 configs
for cfg in [
        Dict("C"=>0.1, "kernel"=>"linear"),
        Dict("C"=>1.0, "kernel"=>"rbf", "gamma"=>0.05),
        Dict("C"=>5.0, "kernel"=>"poly", "degree"=>3, "gamma"=>0.1, "coef0"=>1.0)]
    res = modelCrossValidation(:SVC, cfg, (X, y), indices_cv)
    show_results(cfg, res)
end

# DoME – 3 valores máximo nodos
for nodes in [10, 20, 30]
    cfg = Dict("maximumNodes"=>nodes)
    res = modelCrossValidation(:DoME, cfg, (X, y), indices_cv)
    show_results(cfg, res)
end

# Árbol – 3 profundidades
for depth in [4, 8, 12]
    cfg = Dict("max_depth"=>depth)
    res = modelCrossValidation(:DecisionTreeClassifier, cfg, (X, y), indices_cv)
    show_results(cfg, res)
end

# k‑NN – 3 k
for k_nn in [3, 5, 7]
    cfg = Dict("n_neighbors"=>k_nn)
    res = modelCrossValidation(:KNeighborsClassifier, cfg, (X, y), indices_cv)
    show_results(cfg, res)
end



# GRÁFICAS  

values = [nrow(df_rocks), nrow(df_mines)]
bar([1,1.2], values; fillcolor=RGB(0.69,0,0.45), linecolor=:transparent, legend=false,
    xticks=([1,1.2],["Roca","Mina"]), ylabel="Nº muestras", size=(400,400))
annotate!([(x,v+4,text(string(v),9,:gray)) for (x,v) in zip([1,1.2],values)])
savefig("muestras.png")

mean_rock = vec(mean(Matrix(df_rocks), dims=1)); std_rock = vec(std(Matrix(df_rocks), dims=1))
mean_mine = vec(mean(Matrix(df_mines), dims=1)); std_mine = vec(std(Matrix(df_mines), dims=1))

plot(1:60, mean_rock; ribbon=std_rock, fillalpha=0.2, label="Media Roca ± SD", linewidth=2)
plot!(1:60, mean_mine; ribbon=std_mine, fillalpha=0.2, label="Media Mina ± SD", linewidth=2)
savefig("medias_frecuencia.png")




