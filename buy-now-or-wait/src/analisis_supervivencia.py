from pyspark.sql import Window
from pyspark.sql import functions as F

from config import PARQUET_DIR, get_spark

spark = get_spark("desaparicion")

base = (spark.read.parquet(str(PARQUET_DIR))
        .filter(F.col("flightDate") < F.lit("2022-11-01"))
        .withColumn("sd_num", F.datediff("searchDate", F.lit("2022-01-01"))))

w = Window.partitionBy("legId").orderBy("sd_num")

d = (base
     .withColumn("n_7d", F.count("totalFare").over(w.rangeBetween(1, 7)))
     .withColumn("futuras", F.count("totalFare").over(w.rowsBetween(1, Window.unboundedFollowing)))
     .withColumn("desaparece", ((F.col("futuras") == 0) & (F.col("daysToDeparture") > 7)).cast("int"))
     .withColumn("hueco", ((F.col("n_7d") == 0) & (F.col("futuras") > 0)).cast("int"))
     .filter(F.col("daysToDeparture") > 7)
     .cache())

print("observaciones:", d.count())
d.select(
    F.round(F.mean("desaparece") * 100, 2).alias("pct_desaparicion_terminal"),
    F.round(F.mean("hueco") * 100, 2).alias("pct_huecos"),
).show()

print("¿correlaciona con los asientos restantes?")
d.groupBy("seatsRemaining").agg(
    F.count("*").alias("n"),
    F.round(F.mean("desaparece") * 100, 2).alias("pct_desaparece"),
).orderBy("seatsRemaining").show(12)

print("por antelacion")
d.groupBy((F.floor(F.col("daysToDeparture") / 10) * 10).alias("tramo_dias")).agg(
    F.count("*").alias("n"),
    F.round(F.mean("desaparece") * 100, 2).alias("pct_desaparece"),
).orderBy("tramo_dias").show()

print("por aerolinea")
d.groupBy("firstAirline").agg(
    F.count("*").alias("n"),
    F.round(F.mean("desaparece") * 100, 2).alias("pct_desaparece"),
).filter(F.col("n") > 50000).orderBy(F.desc("pct_desaparece")).show()

w_dia = Window.partitionBy("route", "flightDate", "franja", "isNonStop", "searchDate")
w_7d = (Window.partitionBy("route", "flightDate", "franja", "isNonStop")
        .orderBy("sd_num").rangeBetween(1, 7))

m = (d
     .withColumn("franja", F.floor(F.col("depHour") / 4))
     .withColumn("mejor_hoy", F.min("totalFare").over(w_dia))
     .withColumn("mejor_7d", F.min("mejor_hoy").over(w_7d))
     .withColumn("deterioro", F.col("mejor_7d") - F.col("mejor_hoy"))
     .cache())

print("deterioro de la mejor alternativa comparable en 7 dias")
for etiqueta, cond in [("DESAPARECE", F.col("desaparece") == 1),
                       ("SOBREVIVE", F.col("desaparece") == 0)]:
    print(etiqueta)
    m.filter(cond & F.col("mejor_7d").isNotNull()).select(
        F.count("*").alias("n"),
        F.round(F.mean("deterioro"), 2).alias("deterioro_medio"),
        F.round(F.mean(F.greatest(F.col("deterioro"), F.lit(0.0))), 2).alias("penalizacion_media"),
        F.round(F.mean((F.col("deterioro") > 0).cast("double")) * 100, 2).alias("pct_empeora"),
    ).show()

print("penalizacion esperada por esperar, en $ por busqueda")
m.filter(F.col("mejor_7d").isNotNull()).select(
    F.round(F.mean(F.col("desaparece") * F.greatest(F.col("deterioro"), F.lit(0.0))), 3)
     .alias("penalizacion_esperada_usd"),
).show()

spark.stop()