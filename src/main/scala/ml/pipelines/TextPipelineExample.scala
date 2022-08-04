package ml.pipelines

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.Row
import spark.SparkSessionObject

object TextPipelineExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSessionObject.getSparkSession("EstimatorTransformerParamExample")

    import spark.implicits._

    // Prepare training documents from a list of (id, text, label) tuples.
    val trainingData = spark
      .createDataFrame(
        Seq(
          (0L, "a b c d e spark", 1.0),
          (1L, "b d", 0.0),
          (2L, "spark f g h", 1.0),
          (3L, "hadoop mapreduce", 0.0)
        )
      )
      .toDF("id", "text", "label")

    // Transformer - converts raw text into words (splits it by white spaces)
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    // Transformer - converts the words into feature vectors
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

    // Estimator - statistical analysis method to predict a binary outcome (0 or 1)
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)

    // Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))

    // Fit the pipeline to training documents.
    val model = pipeline.fit(trainingData)

    // Now we can optionally save the fitted pipeline to disk
    model.write.overwrite().save("/tmp/spark-logistic-regression-model")

    // We can also save this unfit pipeline to disk
    pipeline.write.overwrite().save("/tmp/unfit-lr-model")

    // And load it back in during production
    val sameModel = PipelineModel.load("/tmp/spark-logistic-regression-model")

    // Prepare test documents, which are unlabeled (id, text) tuples.
    val testData = spark
      .createDataFrame(
        Seq(
          (4L, "spark i j k"),
          (5L, "l m n"),
          (6L, "spark hadoop spark"),
          (7L, "apache hadoop")
        )
      )
      .toDF("id", "text")

    model
      .transform(testData)
      .select("id", "text", "probability", "prediction")
      .collect()
      .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
        println(s"($id, $text) --> prob=$prob, prediction=$prediction")
      }

    spark.close()

  }

}
