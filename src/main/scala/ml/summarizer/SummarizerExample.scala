package ml.summarizer

import org.apache.spark.ml.linalg.{Vector, Vectors}
import spark.SparkSessionObject

object SummarizerExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSessionObject.getSparkSession("SummarizerExample")

    import org.apache.spark.ml.stat.Summarizer._
    import spark.implicits._

    // $example on$
    val data = Seq(
      (Vectors.dense(2.0, 3.0, 5.0), 1.0),
      (Vectors.dense(4.0, 6.0, 7.0), 2.0)
    )

    val df = data.toDF("features", "weight")

    val (meanVal, varianceVal) = df.select(metrics("mean", "variance")
      .summary($"features", $"weight").as("summary"))
      .select("summary.mean", "summary.variance")
      .as[(Vector, Vector)].first()

    println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")

    val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features"))
      .as[(Vector, Vector)].first()

    println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
