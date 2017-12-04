import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable
import scala.util.parsing.json.{JSON, JSONObject}
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.IntegerType

/**
  * Created by pavel.a.popov on 01.12.17.
  */
object VKPosts {

  val spark = SparkSession.builder.appName("Simple Application").master("local").getOrCreate

  import spark.implicits._

  import scala.util.Try

  def getTrainDaraFrame(path: String, label: Int): DataFrame = {
    val df = spark
      .read
      .json(path)
      .toDF()
      .select("response")
      .filter(json => {
        !json.isNullAt(0)
      })
      .flatMap(json => {
        json.get(0).asInstanceOf[mutable.WrappedArray[String]].map(s => {
          val text = Try {
            JSON.parseRaw(s).get.asInstanceOf[JSONObject].obj.get("text").asInstanceOf[Option[String]].get
          }
          if (text.isSuccess) {
            text.get.trim
          } else {
            ""
          }
        })
      })
      .filter(text => {
        !text.isEmpty
      })
      .toDF("text")
      .withColumn("label", lit(label).cast(IntegerType))
    df
  }
}
