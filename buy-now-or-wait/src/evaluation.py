from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)
from config import RESULTS_DIR


def metricas(pred, prob_col="probability"):
    binario = BinaryClassificationEvaluator(rawPredictionCol=prob_col, labelCol="label")
    multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    return {
        "auc_pr": binario.setMetricName("areaUnderPR").evaluate(pred),
        "auc_roc": binario.setMetricName("areaUnderROC").evaluate(pred),
        "f1": multi.setMetricName("f1").evaluate(pred),
        "accuracy": multi.setMetricName("accuracy").evaluate(pred),
        "precision_esperar": multi.setMetricName("precisionByLabel")
                                  .setMetricLabel(1.0).evaluate(pred),
        "recall_esperar": multi.setMetricName("recallByLabel")
                               .setMetricLabel(1.0).evaluate(pred),
    }


def matriz_confusion(pred):
    celdas = pred.groupBy("label", "prediction").count().collect()
    d = {(int(r["label"]), int(r["prediction"])): r["count"] for r in celdas}
    return {
        "tn": d.get((0, 0), 0), "fp": d.get((0, 1), 0),
        "fn": d.get((1, 0), 0), "tp": d.get((1, 1), 0),
    }

import pandas as pd
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

UMBRALES = [round(u / 100, 2) for u in range(5, 100, 5)]


def _con_prob(pred, prob_col="probability"):
    return pred.withColumn("p1", vector_to_array(F.col(prob_col))[1])


def barrido_umbral(pred, umbrales=UMBRALES, prob_col="probability"):
    p = _con_prob(pred, prob_col).cache()
    ganancia = F.col("totalFare") - F.col("min_7d")

    filas = []
    for u in umbrales:
        esperar = F.col("p1") >= u
        r = p.select(
            F.sum((esperar & (F.col("label") == 1)).cast("int")).alias("tp"),
            F.sum((esperar & (F.col("label") == 0)).cast("int")).alias("fp"),
            F.sum((~esperar & (F.col("label") == 1)).cast("int")).alias("fn"),
            F.sum((~esperar & (F.col("label") == 0)).cast("int")).alias("tn"),
            F.sum(F.when(esperar, ganancia).otherwise(0.0)).alias("ahorro_total"),
            F.count("*").alias("n"),
        ).first()
        filas.append({
            "u": u, "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
            "ahorro_total": r["ahorro_total"], "n": r["n"],
        })
    p.unpersist()

    t = pd.DataFrame(filas)
    t["precision"] = t["tp"] / (t["tp"] + t["fp"])
    t["recall"] = t["tp"] / (t["tp"] + t["fn"])
    t["f1"] = 2 * t["precision"] * t["recall"] / (t["precision"] + t["recall"])
    t["ahorro_por_busqueda"] = t["ahorro_total"] / t["n"]
    return t


def mejor_umbral(tabla, criterio="ahorro_por_busqueda"):
    return float(tabla.loc[tabla[criterio].idxmax(), "u"])

def mostrar(tabla, titulo=""):
    if titulo:
        print(f"\n--- {titulo} ---")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(tabla.to_string(index=False))


def guardar(tabla, nombre):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = RESULTS_DIR / f"{nombre}.csv"
    tabla.to_csv(ruta, index=False)
    print(f"guardado en {ruta}")
    return ruta


def comparar(informes, nombre="comparativa_modelos"):
    t = pd.DataFrame(informes).sort_values("auc_pr", ascending=False)
    mostrar(t, "comparativa de modelos")
    guardar(t, nombre)
    return t

def aplicar_umbral(pred, umbral, prob_col="probability"):
    return (_con_prob(pred, prob_col)
            .withColumn("prediction", (F.col("p1") >= umbral).cast("double")))


def informe(pred, nombre, umbral=0.5):
    p = aplicar_umbral(pred, umbral).cache()
    ganancia = F.col("totalFare") - F.col("min_7d")

    r = p.select(
        F.sum(F.when(F.col("prediction") == 1.0, ganancia).otherwise(0.0)).alias("ahorro"),
        F.sum(ganancia).alias("ahorro_esperar_siempre"),
        F.count("*").alias("n"),
    ).first()

    fila = {"modelo": nombre, "umbral": umbral}
    fila.update({k: round(v, 4) for k, v in metricas(p).items()})
    fila.update(matriz_confusion(p))
    fila["ahorro_modelo"] = round(r["ahorro"] / r["n"], 2)
    fila["ahorro_esperar_siempre"] = round(r["ahorro_esperar_siempre"] / r["n"], 2)
    fila["ahorro_comprar_siempre"] = 0.0

    p.unpersist()
    return fila


def pintar_barrido(tabla, titulo="Barrido de umbral"):
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(tabla["u"], tabla["precision"], label="precisión")
    ax1.plot(tabla["u"], tabla["recall"], label="recall")
    ax1.plot(tabla["u"], tabla["f1"], label="F1")
    ax1.set_xlabel("umbral")
    ax1.set_ylabel("métrica")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(tabla["u"], tabla["ahorro_por_busqueda"], color="black", ls="--")
    ax2.set_ylabel("ahorro por búsqueda ($)")

    plt.title(titulo)
    plt.show()

def valor_decision(pred, umbral, prob_col="probability"):
    p = aplicar_umbral(pred, umbral, prob_col)
    ganancia = F.col("totalFare") - F.col("min_7d")
    comprar = F.col("prediction") == 0.0

    r = p.select(
        F.sum(comprar.cast("int")).alias("n_comprar"),
        F.sum(F.when(comprar, F.greatest(-ganancia, F.lit(0.0)))).alias("evitada"),
        F.sum(F.when(comprar, F.greatest(ganancia, F.lit(0.0)))).alias("renunciado"),
        F.count("*").alias("n"),
    ).first()

    n_comprar = max(r["n_comprar"], 1)
    return pd.DataFrame([{
        "umbral": umbral,
        "n_comprar_ya": r["n_comprar"],
        "pct_comprar_ya": round(100 * r["n_comprar"] / r["n"], 2),
        "perdida_evitada": round(r["evitada"], 2),
        "ahorro_renunciado": round(r["renunciado"], 2),
        "neto_por_recomendacion": round((r["evitada"] - r["renunciado"]) / n_comprar, 2),
    }])
    