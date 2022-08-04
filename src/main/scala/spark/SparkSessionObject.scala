package spark

import org.apache.spark.sql.SparkSession

object SparkSessionObject {

  def getSparkSession(appName : String, logLevel : String = "WARN"): SparkSession = {
    val spark = SparkSession
      .builder
      .appName(appName)
      .config("spark.master", "local")
      .getOrCreate()

    spark.sparkContext.setLogLevel(logLevel)

    spark

  }

}
