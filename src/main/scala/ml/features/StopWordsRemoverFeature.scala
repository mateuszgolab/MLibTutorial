package ml.features

import org.apache.spark.ml.feature.StopWordsRemover
import spark.SparkSessionObject

object StopWordsRemoverFeature {

  def main(args: Array[String]): Unit = {
    val spark = SparkSessionObject.getSparkSession("StopWordsRemover")

    // english (default)
    val remover = new StopWordsRemover()
      .setInputCol("raw")
      .setOutputCol("filtered")

    // italian
    val italianStopWords = StopWordsRemover.loadDefaultStopWords("italian")
    val italianRemover = new StopWordsRemover()
      .setOutputCol("filtered")
      .setInputCol("raw")
      .setStopWords(italianStopWords)


    val dataSet = spark.createDataFrame(Seq(
      (0, Seq("I", "saw", "the", "red", "balloon")),
      (1, Seq("Mary", "had", "a", "little", "lamb"))
    )).toDF("id", "raw")

    println("English (default)")
    remover
      .transform(dataSet)
      .show(false)

    println("Italian")
    italianRemover
      .transform(dataSet)
      .show(false)

    spark.close()

  }

}
