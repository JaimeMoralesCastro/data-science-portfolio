import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# si la variable del sistema no vale, tiramos de la ruta nuestra
JAVA_HOME = os.environ.get("JAVA_HOME", "")
if not Path(JAVA_HOME).is_dir():
    JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"

HADOOP_HOME = os.environ.get("HADOOP_HOME", r"C:\hadoop")
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "itineraries.csv"
PARQUET_DIR = DATA_DIR / "corredor.parquet"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
SUBSET_CSV = DATA_DIR / "corredor_transcontinental.csv"


ORIGINS = ["LAX", "SFO"]
DESTINATIONS = ["JFK", "EWR", "BOS"]
AIRPORTS = ORIGINS + DESTINATIONS

SEED = 42
DROP_MARGIN = 0.05
TEST_FRACTION = 0.20
VAL_FRACTION = 0.15
N_CORES = 8
DRIVER_MEMORY = "8g"
SHUFFLE_PARTITIONS = 48


def _setup_env():
    if Path(JAVA_HOME).is_dir():
        os.environ["JAVA_HOME"] = JAVA_HOME
    # winutils solo hace falta en windows
    if os.name == "nt" and (Path(HADOOP_HOME) / "bin" / "winutils.exe").is_file():
        os.environ["HADOOP_HOME"] = HADOOP_HOME
        os.environ["PATH"] = str(Path(HADOOP_HOME) / "bin") + os.pathsep + os.environ["PATH"]
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark(app_name="AAGE-P1", cores=N_CORES, memory=DRIVER_MEMORY):
    _setup_env()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .master(f"local[{cores}]")
        .appName(app_name)
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    for d in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return spark


if __name__ == "__main__":
    spark = get_spark("config-check")
    jvm = spark.sparkContext._jvm
    print("spark:", spark.version)
    print("python:", sys.executable)
    print("heap driver (GB):", round(jvm.java.lang.Runtime.getRuntime().maxMemory() / 1024**3, 2))
    print("csv origen existe:", RAW_CSV.is_file())
    spark.stop()