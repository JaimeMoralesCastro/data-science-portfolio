import argparse

from pyspark.sql import functions as F

from config import get_spark, SEED
from data_processing import MODELLED_PARQUET
from evaluation import (barrido_umbral, comparar, guardar, informe,
                        mejor_umbral, mostrar, valor_decision)
from models import entrenar, importancias, mejor, predecir, resumen


def cargar(spark, ruta=MODELLED_PARQUET, muestra=None):
    df = spark.read.parquet(str(ruta))
    if muestra:
        df = df.sample(fraction=muestra, seed=SEED)
    partes = [df.filter(F.col("particion") == p).cache()
              for p in ("train", "validation", "test")]
    for nombre, parte in zip(("train", "validation", "test"), partes):
        print(f"{nombre}: {parte.count()} filas")
    return partes


def main(muestra=None):
    spark = get_spark("main")
    try:
        train, val, test = cargar(spark, muestra=muestra)

        resultados = entrenar(train, val)
        mostrar(resumen(resultados), "busqueda de hiperparametros")

        elegido = mejor(resultados)
        print(f"\nmodelo elegido: {elegido['nombre']} {elegido['params']}")

        informes = []
        for r in resultados:
            tabla = barrido_umbral(predecir(r, val))
            u = mejor_umbral(tabla)
            informes.append(informe(predecir(r, test), r["nombre"], u))

            if r["nombre"] == elegido["nombre"]:
                mostrar(tabla, f"barrido de umbral · {r['nombre']}")
                guardar(tabla, "barrido_umbral")
                informes.append(informe(predecir(r, test), r["nombre"] + "_u0.5", 0.5))

        comparar(informes)

        imp = importancias(elegido)
        mostrar(imp, "importancia de variables")
        guardar(imp, "importancia_variables")
        u_elegido = mejor_umbral(barrido_umbral(predecir(elegido, val)))
        mostrar(valor_decision(predecir(elegido, test), u_elegido),
                "valor de las recomendaciones de COMPRAR YA")

    finally:
        spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--muestra", type=float, default=None,
                   help="fraccion del dataset, p.ej. 0.05 para una prueba rapida")
    main(p.parse_args().muestra)