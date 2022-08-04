package ml.features

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.functions.{col, udf}
import spark.SparkSessionObject

object WordsTokenizer {

  def main(args: Array[String]): Unit = {
    val spark = SparkSessionObject.getSparkSession("WordsTokenizer")

    val sentenceDataFrame = spark.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (1, "I wish Java could use case classes"),
      (2, "Logistic,regression,models,are,neat")
    )).toDF("id", "sentence")

    val tokenizer = new Tokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")

    val regexTokenizer = new RegexTokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

    val countWords = udf { (words: Seq[String]) => words.length }

    println("Simple Tokenizer")
    val tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized
      .select("sentence", "words")
      .withColumn("word count", countWords(col("words")))
      .show(false)

    println("Regex Tokenizer")
    val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized
      .select("sentence", "words")
      .withColumn("word count", countWords(col("words")))
      .show(false)


    spark.close()

  }

}
