package classifier

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.IntegerType

/**
  * Created by pavel.a.popov on 06.12.17.
  */
object Classifier extends App {
  val spark = SparkSession.builder.appName("Simple Application").master("local[4]").getOrCreate

  import spark.implicits._

  try {

    val train1 = spark
      .createDataset(Seq(
        "ein zwei drei",
        "und mir sicht",
        "ich liebe dich",
        "main hast kleine"
      ))
      .toDF("text")
      .withColumn("label", lit(0).cast(IntegerType))

    val train2 = spark
      .createDataset(Seq(
        "one two three",
        "i love you",
        "it is not my opinion",
        "my little house thinking think about it"
      ))
      .toDF("text")
      .withColumn("label", lit(1).cast(IntegerType))

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      //.setNumFeatures(2000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val lr = new NaiveBayes()
    //val lr = new LogisticRegression()

    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

    val model = pipeline.fit(train1.union(train2))

    val test = spark
      .createDataset(Seq(
        "ein zwei drei",
        "und mir sicht",
        "one my house",
        "i love my zwei house",
        "main hast about zwei little kleine house"
      ))
      .toDF("text")

    model.transform(test)
      .select("text", "features", "prediction", "rawPrediction")
      .collect()
      .foreach { r =>
        println(s"(prediction=${r.getAs("prediction")}, features=${r.getAs("features")},rawPrediction=${r.getAs("rawPrediction")}")
      }

  } finally {
    spark.stop()
  }

}
