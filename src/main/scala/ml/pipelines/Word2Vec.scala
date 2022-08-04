package ml.pipelines

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import spark.SparkSessionObject

object Word2Vec {

  def main(args: Array[String]): Unit = {
    val spark = SparkSessionObject.getSparkSession("EstimatorTransformerParamExample")

    // Input data: Each row is a bag of words from a sentence or document.
    val documentDF = spark
      .createDataFrame(
        Seq(
          "Hi I heard about Spark".split(" "),
          "I wish Java could use case classes".split(" "),
          "Logistic regression models are neat".split(" ")
        ).map(Tuple1.apply)
      )
      .toDF("text")

    documentDF.show(false)

    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)

    val model = word2Vec.fit(documentDF)

    val result = model.transform(documentDF)
    result.collect().foreach { case Row(text: Array[String], features: Vector) =>
      println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n")
    }

    spark.close()

  }

}
