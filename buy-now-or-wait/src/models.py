import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (DecisionTreeClassifier, GBTClassifier,
                                       LogisticRegression, RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (Imputer, OneHotEncoder, StandardScaler,
                                StringIndexer, VectorAssembler)
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import functions as F

from config import SEED

NUMERICAS = [
    "daysToDeparture", "totalFare", "baseFare", "precio_rel", "seatsRemaining",
    "durationMinutes", "totalTravelDistance", "nSegments", "nAirlines",
    "depHour", "elapsedDays", "dow_vuelo", "dow_busqueda",
    "n_obs_previas", "fare_anterior", "var_anterior", "fare_primera",
    "var_desde_primera", "min_historico", "media_historica",
    "precio_vs_min_historico",
    "mercado_n", "mercado_min", "mercado_std", "precio_percentil",
    "precio_vs_mercado_min", "mercado_momento",
]

CATEGORICAS = ["route", "firstAirline", "firstCabin", "claseTarifaria"]

BOOLEANAS = ["isBasicEconomy", "isNonStop", "isAlternateAirport",
             "es_primera_obs", "asientos_desconocido"]

IMPUTADAS = [c + "_imp" for c in NUMERICAS]
INDICES = [c + "_idx" for c in CATEGORICAS]


def preparar(df):
    for c in NUMERICAS:
        df = df.withColumn(c, F.col(c).cast("double"))
    for c in BOOLEANAS:
        df = df.withColumn(c, F.col(c).cast("int"))
    return df


def _comunes():
    return [
        Imputer(inputCols=NUMERICAS, outputCols=IMPUTADAS, strategy="median"),
        StringIndexer(inputCols=CATEGORICAS, outputCols=INDICES, handleInvalid="keep"),
    ]


def etapas_arboles():
    return _comunes() + [
        VectorAssembler(inputCols=IMPUTADAS + BOOLEANAS + INDICES,
                        outputCol="features", handleInvalid="error")
    ]


def etapas_lineal():
    return _comunes() + [
        OneHotEncoder(inputCols=INDICES, outputCols=[c + "_ohe" for c in CATEGORICAS]),
        VectorAssembler(inputCols=IMPUTADAS + BOOLEANAS + [c + "_ohe" for c in CATEGORICAS],
                        outputCol="features_raw", handleInvalid="error"),
        StandardScaler(inputCol="features_raw", outputCol="features",
                       withMean=True, withStd=True),
    ]


def catalogo():
    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50)
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", seed=SEED,
                                maxBins=32, maxMemoryInMB=64)
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=SEED,
                                maxBins=32, maxMemoryInMB=64)
    gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=SEED,
                        maxBins=32, maxMemoryInMB=64)

    return [
        ("logistica", "lineal", lr,
         ParamGridBuilder()
         .addGrid(lr.regParam, [0.0, 0.01, 0.1])
         .addGrid(lr.elasticNetParam, [0.0, 0.5])
         .build()),

        ("arbol", "arboles", dt,
         ParamGridBuilder()
         .addGrid(dt.maxDepth, [5, 10, 15])
         .build()),

        ("random_forest", "arboles", rf,
         ParamGridBuilder()
         .addGrid(rf.numTrees, [50, 100])
         .addGrid(rf.maxDepth, [8, 12])
         .build()),

        ("gbt", "arboles", gbt,
         ParamGridBuilder()
         .addGrid(gbt.maxIter, [50, 100])
         .addGrid(gbt.maxDepth, [4, 6])
         .build()),
    ]


def _describir(params):
    return {p.name: v for p, v in params.items()}


def entrenar(train, val):
    train, val = preparar(train), preparar(val)
    evaluador = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderPR")

    etapas = {"lineal": etapas_lineal, "arboles": etapas_arboles}
    resultados = []

    for tipo in ("lineal", "arboles"):
        prep = Pipeline(stages=etapas[tipo]()).fit(train)
        tr = prep.transform(train).select("features", "label").cache()
        va = prep.transform(val).cache()

        for nombre, t, estimador, grid in catalogo():
            if t != tipo:
                continue

            mejor_m, mejor_auc, mejor_params = None, -1.0, None
            for i, params in enumerate(grid, 1):
                m = estimador.fit(tr, params)
                auc = evaluador.evaluate(m.transform(va))
                print(f"{nombre} [{i}/{len(grid)}] auc_pr={auc:.4f} {_describir(params)}")
                if auc > mejor_auc:
                    mejor_m, mejor_auc, mejor_params = m, auc, params

            resultados.append({
                "nombre": nombre,
                "modelo": PipelineModel(stages=prep.stages + [mejor_m]),
                "auc_pr_val": round(mejor_auc, 4),
                "params": _describir(mejor_params),
            })
            print(f"--> mejor {nombre}: auc_pr={mejor_auc:.4f} {_describir(mejor_params)}\n")

        tr.unpersist()
        va.unpersist()

    return resultados


def mejor(resultados):
    return max(resultados, key=lambda r: r["auc_pr_val"])


def predecir(resultado, df):
    return resultado["modelo"].transform(preparar(df))


def importancias(resultado, top=20):
    final = resultado["modelo"].stages[-1]
    pesos = final.featureImportances.toArray()
    nombres = IMPUTADAS + BOOLEANAS + INDICES

    t = pd.DataFrame({"variable": nombres, "peso": pesos})
    return t.sort_values("peso", ascending=False).head(top).reset_index(drop=True)


def resumen(resultados):
    return pd.DataFrame([
        {"modelo": r["nombre"],
         "auc_pr_val": r["auc_pr_val"],
         "params": str(r["params"])}
        for r in resultados
    ]).sort_values("auc_pr_val", ascending=False).reset_index(drop=True)
