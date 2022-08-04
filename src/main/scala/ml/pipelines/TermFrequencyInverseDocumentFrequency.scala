package ml.pipelines

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import spark.SparkSessionObject

object TermFrequencyInverseDocumentFrequency {

  def main(args: Array[String]): Unit = {
    val spark = SparkSessionObject.getSparkSession("EstimatorTransformerParamExample")

    import spark.implicits._

    val sentenceData = spark
      .createDataFrame(
        Seq(
          (0.0, "Hi I heard about Spark"),
          (0.0, "I wish Java could use case classes"),
          (1.0, "Logistic regression models are neat")
        )
      )
      .toDF("label", "sentence")

    val tokenizer = new Tokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")

    val wordsData = tokenizer.transform(sentenceData)

    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(20)

    val featurizedData = hashingTF.transform(wordsData)

    /**
     * IDF is an Estimator which is fit on a dataset and produces an IDFModel.
     * The IDFModel takes feature vectors (generally created from HashingTF or CountVectorizer) and scales each feature.
     * Intuitively, it down-weights features which appear frequently in a corpus.
     */
    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")

    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)

    rescaledData
      .select("label", "features")
      .show(false)

  }

}
