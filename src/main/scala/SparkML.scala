import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{SparkSession}

case class Features(s:String)


object SparkML extends App {

  val spark = SparkSession.builder.appName("Simple Application").master("local").getOrCreate
  import spark.implicits._
  val colNos = Seq(0, 3, 5)
  val df1 = spark
    .read
    .option("delimiter", ";")
    .option("header", "true")
    .csv("data/*1.csv")
    .map(row=>{
      Features("d")
    })

  df1.show()

  val training = spark.createDataFrame(Seq(
    (1.0, Vectors.dense(0.1, 1.1, 2.1)),
    (1.0, Vectors.dense(0.0, 1.3, 2.1)),
    (1.0, Vectors.dense(0.0,1.3, 2.2)),
    (1.0, Vectors.dense(0.9, 1.1, 0.1)),
    (2.0, Vectors.dense(2.0, 1.0, 1.0)),
    (2.0, Vectors.dense(2.0, 3.2, 0.5))
  )).toDF("label", "features")

  val lr = new LogisticRegression()
  println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

  lr.setMaxIter(10)
    .setRegParam(0.01)

  val model1 = lr.fit(training)
  println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)

  val paramMap = ParamMap(lr.maxIter -> 20)
    .put(lr.maxIter, 30)
    .put(lr.regParam -> 0.1, lr.threshold -> 0.55)

  val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")
  val paramMapCombined = paramMap ++ paramMap2

  val model2 = lr.fit(training, paramMapCombined)
  println("Model 2 was fit using parameters: " + model2.parent.extractParamMap)

  val test = spark.createDataFrame(Seq(
    (0.0,Vectors.dense(1.0, 1.5, 1.3)),
    (0.0,Vectors.dense(0.0, 1.5, 1.3)),
    (0.0,Vectors.dense(1.5, 1.5, 1.3)),
    (0.0,Vectors.dense(1.0, 1.5, 1.3))
  )).toDF("label", "features")

  model2.transform(test)
    .select("features", "label", "myProbability", "prediction")
    .collect()
    .foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
      println(s"($features, $label) -> prob=$prob, prediction=$prediction")
    }
println(lr.explainParams())
  spark.stop()
}