package ml.features

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import spark.SparkSessionObject

object CountVectorizer {

  def main(args: Array[String]): Unit = {
    val spark = SparkSessionObject.getSparkSession("EstimatorTransformerParamExample")

    val df = spark
      .createDataFrame(
        Seq(
          (0, Array("a", "b", "c")),
          (1, Array("a", "b", "b", "c", "a"))
        )
      )
      .toDF("id", "words")

    /**
     * CountVectorizer can be used as an Estimator to extract the vocabulary, and generates a CountVectorizerModel.
     * The model produces sparse representations for the documents over the vocabulary,
     * which can then be passed to other algorithms like LDA.
     */
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(df)

    // alternatively, define CountVectorizerModel with a-priori vocabulary
    val cvm = new CountVectorizerModel(Array("a", "b", "c"))
      .setInputCol("words")
      .setOutputCol("features")

//    Each vector represents the token counts of the document over the vocabulary.
    cvModel.transform(df).show(false)

    spark.close()

  }

}
