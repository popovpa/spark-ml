import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.sql.{Row, SparkSession}

/**
  * Created by pavel.a.popov on 22.11.17.
  */
object TextClassificationML  extends App{

  val spark = SparkSession.builder.appName("Simple Application").master("local").getOrCreate
  val training = spark.createDataFrame(Seq(
    ("buy now sale enlarge", 1.0),
    ("only today your save money", 1.0),
    ("online shop sale now", 1.0),
    ("we want to buy you", 0.0),
    ("we discuss about dealing", 0.0),
    ("business sale money", 0.0),
    ("trip money road journey", 2.0),
    ("road drive journey", 2.0),
    ("trip money road drive", 2.0)
  )).toDF("text", "label")

  val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")
  val hashingTF = new HashingTF()
    .setNumFeatures(1000)
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("features")
  val lr = new NaiveBayes()

  val pipeline = new Pipeline()
    .setStages(Array(tokenizer, hashingTF, lr))

  val model = pipeline.fit(training)

  model.write.overwrite().save("tmp/spark-logistic-regression-model")

  pipeline.write.overwrite().save("tmp/unfit-lr-model")

  val sameModel = PipelineModel.load("tmp/spark-logistic-regression-model")

  val test = spark.createDataFrame(Seq(
    (4L, "enlarge your money"),
    (4L, "business shop now"),
    (4L, "sale today"),
    (4L, "sale discuss"),
    (0L, "journey sale trip")
  )).toDF("id", "text")

  model.transform(test)
    .select("text", "probability", "prediction")
    .collect()
    .foreach { r =>
      println(s"(${r.getAs("id")}, ${r.getAs("text")}) --> prob=${r.getAs("probability")}, prediction=${r.getAs("prediction")}")
    }


}
