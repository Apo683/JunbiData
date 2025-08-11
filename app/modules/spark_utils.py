from pyspark.sql import SparkSession
import time

# Variable globale pour stocker la SparkSession
_spark_session = None

def get_spark_session():
    """Récupère ou crée une SparkSession globale."""
    global _spark_session
    if _spark_session is None:
        start_time = time.time()
        _spark_session = SparkSession.builder.appName("JunbiData").master("local[*]").getOrCreate()
        elapsed_time = time.time() - start_time
        print(f"SparkSession créée en {elapsed_time:.2f} secondes")
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