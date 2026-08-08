# app/utils/spark_utils.py
from pyspark.sql import SparkSession
import sys
import tempfile
import os
import time
from multiprocessing import cpu_count

_spark_session = None

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default

def _is_wsl() -> bool:
    # Heuristique simple
    return "WSL_INTEROP" in os.environ or "WSL_DISTRO_NAME" in os.environ

def _default_local_dirs() -> str:
    # Évite /mnt/c|d (NTFS) sous WSL → préfère le FS Linux (ext4)
    if _is_wsl():
        base = os.path.expanduser("~/spark-tmp")
    else:
        base = os.path.join(tempfile.gettempdir(), "spark-tmp")

    os.makedirs(base, exist_ok=True)
    return base

def get_spark_session():
    """
    Récupère ou crée une SparkSession globale, configurée pour limiter les OOM en local.
    Les principaux réglages sont overridables via variables d'environnement:

    JUNBI_SPARK_DRIVER_MEMORY=24g
    JUNBI_SPARK_DRIVER_MAX_RESULT_SIZE=2g  (0 = illimité, déconseillé)
    JUNBI_SPARK_SHUFFLE_PARTITIONS=auto    (ou un entier ex: 400)
    JUNBI_SPARK_LOCAL_DIRS=/path1,/path2
    JUNBI_SPARK_BCAST_THRESHOLD=50m        (autoBroadcastJoinThreshold)
    JUNBI_SPARK_ARROW_BATCH=20000
    JUNBI_SPARK_MASTER=local[*]
    JUNBI_SPARK_LOGLEVEL=WARN
    """
    global _spark_session
    if _spark_session is not None:
        return _spark_session

    start_time = time.time()

    # Defaults raisonnables
    master = _env("JUNBI_SPARK_MASTER", "local[*]")
    driver_mem = _env("JUNBI_SPARK_DRIVER_MEMORY", "8g")
    max_result = _env("JUNBI_SPARK_DRIVER_MAX_RESULT_SIZE", "2g")  # "0" pour illimité
    arrow_batch = int(_env("JUNBI_SPARK_ARROW_BATCH", "20000"))
    log_level = _env("JUNBI_SPARK_LOGLEVEL", "WARN")
    bcast_th = _env("JUNBI_SPARK_BCAST_THRESHOLD", "50m")
    local_dirs = _env("JUNBI_SPARK_LOCAL_DIRS", _default_local_dirs())

    # Partitions de shuffle: 4x nb cœurs par défaut (limite la taille des blocs)
    if _env("JUNBI_SPARK_SHUFFLE_PARTITIONS", "auto") == "auto":
        try:
            cores = cpu_count()
        except Exception:
            cores = 8
        shuffle_partitions = str(max(200, 4 * cores))  # borne plancher
    else:
        shuffle_partitions = _env("JUNBI_SPARK_SHUFFLE_PARTITIONS", "400")

    python_executable = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable

    builder = (
        SparkSession.builder
        .appName("JunbiData")
        .master(master)
        # Même Python pour le driver et les workers
        .config("spark.pyspark.python", python_executable)
        .config("spark.pyspark.driver.python", python_executable)
        # Diagnostic des crashs Python
        .config("spark.python.worker.faulthandler.enabled", "true")
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        # Mémoire / résultats
        .config("spark.driver.memory", driver_mem)
        .config("spark.driver.maxResultSize", max_result)
        # Shuffle / parallélisme
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.default.parallelism", shuffle_partitions)
        # AQE et skew
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        # Broadcast joins
        .config("spark.sql.autoBroadcastJoinThreshold", bcast_th)
        # Arrow (Pandas interop) + batch pour éviter pics mémoire
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", str(arrow_batch))
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        # Serializer plus efficient
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # Taille des partitions fichiers (réduit la pression mémoire lors des scans)
        .config("spark.sql.files.maxPartitionBytes", str(128 * 1024 * 1024))  # 128MB
        # Mémoire interne Spark (optionnel, garde les défauts si tu préfères)
        # .config("spark.memory.fraction", "0.6")
        # .config("spark.memory.storageFraction", "0.3")
        # Répertoires pour shuffle / spill
        .config("spark.local.dir", local_dirs)
    )

    _spark_session = builder.getOrCreate()
    # Log level réduit
    _spark_session.sparkContext.setLogLevel(log_level)

    # Imprime un récap utile au démarrage
    sc = _spark_session.sparkContext
    try:
        cores = sc.defaultParallelism
    except Exception:
        cores = "n/a"
    print(f"Python Spark : {python_executable}")
    print(
        f"[Spark] master={master} cores≈{cores} driverMemory={driver_mem} "
        f"shufflePartitions={shuffle_partitions} arrowBatch={arrow_batch} "
        f"localDirs={local_dirs}"
    )
    print(f"SparkSession créée en {time.time() - start_time:.2f} secondes")
    print("Spark :", _spark_session.version)
    print(
        "Hadoop utilisé par Spark :",
        _spark_session.sparkContext._jvm
            .org.apache.hadoop.util.VersionInfo
            .getVersion()
    )
    print(
        "Java utilisé par Spark :",
        _spark_session.sparkContext._jvm.java.lang.System
            .getProperty("java.version")
    )
    return _spark_session


def stop_spark_session():
    """Arrête la SparkSession globale si elle existe."""
    global _spark_session
    if _spark_session is not None:
        _spark_session.stop()
        _spark_session = None
        print("SparkSession arrêtée")


def is_spark_active():
    """Vérifie si une SparkSession est active."""
    return _spark_session is not None
