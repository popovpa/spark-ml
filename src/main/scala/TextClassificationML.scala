import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

/**
  * Created by pavel.a.popov on 22.11.17.
  */
object TextClassificationML extends App {

  val spark = SparkSession.builder.appName("Simple Application").master("local").getOrCreate

  import spark.implicits._

  val Array(df1, test1) = VKPosts.getTrainDaraFrame("data/text/g1/100*.txt", 0).randomSplit(Array(0.9, 0.1))
  //val Array(df2, test2) = VKPosts.getTrainDaraFrame("data/text/g2/100*.txt", 1).randomSplit(Array(0.9, 0.1))
  //val Array(df3, test3) = VKPosts.getTrainDaraFrame("data/text/g3/100*.txt", 2).randomSplit(Array(0.9, 0.1))

  var training = Seq.empty[(String, Int)].toDF("text", "label")

  training = training.union(df1)
 // training = training.union(df2)
 // training = training.union(df3)

  val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")
  val hashingTF = new HashingTF()
    .setNumFeatures(2000)
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("features")
  //val lr = new NaiveBayes()
  val lr = new LogisticRegression()
  //  val lr = new GBTClassifier()
  //    .setLabelCol("label")
  //    .setFeaturesCol("words")
  //    .setMaxIter(10)

  val pipeline = new Pipeline()
    .setStages(Array(tokenizer, hashingTF, lr))

  val model = pipeline.fit(training)

  model.write.overwrite().save("tmp/spark-logistic-regression-model")

  pipeline.write.overwrite().save("tmp/unfit-lr-model")

  val sameModel = PipelineModel.load("tmp/spark-logistic-regression-model")

  val test = spark.createDataFrame(Seq(
    //1
    (4L, "СВОБОДНАЯ МАРЫСЯ! Я иногда бываю удивительно своевременна. С сегодняшнего вечера я в Мск. Обратно в понедельник или вторник вечером.  2. Доступна для встреч, начиная со второй половины воскресенья.  3. Не хочет ли кто-нибудь вписать меня с вс на пн и потенциально с пн на вт?  4. Собственно если на вторник придумаются планы - поеду обратно во вторник. Если не придумаются - в понедельник.  Понять надо до завтрашнего утра, чтобы успеть купить билет обратный до отъезда на полигон."),
    (4L, "Правильно Неправильно Смотрим Запоминаем"),
    (4L, "Новая коллекция белых портупей В наличии в нашем шоуруме"),
    (4L, "Друзья! Я учусь на 3 курсе педагогического колледжа и сейчас активно ищу ученика  1–6 класса. Владею всеми стандартными школьными предметами. ( Французский, древнегреческий и кибербезопасность в них не входят☺)   Могу забирать с доп. занятий и делать домашние задания."),

    (4L, "Бухта Афродиты. Куча поверий, легенд и мой традиционный заплыв в гордом одиночестве, пока остальные решаются разве что потрогать водичку ногами (она, кстати, тёплая). Ну, а я выхожу из пены морской, так сказать. Не могла позволить себе упустить такую возможность."),
    (4L, "Так выглядят большинство мужиков, когда обижаются, и психуют, как бабы."),
    (4L, "Внимание, конкурс! Сегодня мы вместе с нашими партнерами, магазином Чай Тянь Жень www.tianren.ru , начинаем конкурс Лето с чаем. Получить ценный и полезный приз – очень")
  )).toDF("id", "text")

  model.transform(test1)
    .select("text", "features", "prediction","rawPrediction")
    .collect()
    .foreach { r =>
      println(s"(prediction=${r.getAs("prediction")}, features=${r.getAs("features")},rawPrediction=${r.getAs("rawPrediction")}")
    }


}
