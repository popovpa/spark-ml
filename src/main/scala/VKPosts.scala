import org.apache.spark.sql.SparkSession

import scala.collection.mutable
import scala.util.parsing.json.{JSON, JSONObject}

/**
  * Created by pavel.a.popov on 01.12.17.
  */
object VKPosts extends App {

  val spark = SparkSession.builder.appName("Simple Application").master("local").getOrCreate

  import spark.implicits._

  import scala.util.Try

  val df = spark
    .read
    .json("data/text/g1/101*.txt")
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
    }).filter(text => {
    !text.isEmpty
  })

  df.show(10)
  df.printSchema()
  println(df.count())

}
