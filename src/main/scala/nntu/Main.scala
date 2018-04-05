package nntu

import java.util
import java.util.Random

import smile.data.{Attribute, DataFrame, DateAttribute, NumericAttribute, StringAttribute}
import smile.nlp._
import smile.read
import smile.regression._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scalaz.Scalaz._

object Main extends App {
  val dateAttrName = "Date"
  val appNameAttrName = "AppName"
  val languageAttrName = "Language"
  val versionAttrName = "Version"
  val ratingAttrName = "Rating"
  val titleAttrName = "Title"
  val reviewAttrName = "Review"

  val dataSetAttributes: Array[Attribute] = Array(
    new DateAttribute(dateAttrName, "Date attribute", "MM/dd/yyyy"),
    new StringAttribute(appNameAttrName),
    new StringAttribute(languageAttrName),
    new StringAttribute(versionAttrName),
    new NumericAttribute(ratingAttrName),
    new StringAttribute(titleAttrName),
    new StringAttribute(reviewAttrName)
  )

  val dataSet: DataFrame = DataFrame(read.table(
    file = "src/main/resources/app_review_rating_train_1.csv",
    attributes = dataSetAttributes,
    response = None,
    comment = "%",
    missing = "?",
    delimiter = "\\|",
    header = true,
    rowNames = false
  ))

  val testDataSize = 1000
  val analyseDataSize = 4000

  val beginTestData = new Random(System.currentTimeMillis()).nextInt(dataSet.size - (testDataSize + analyseDataSize) - 2)
  val endTestData = beginTestData + testDataSize
  val beginAnalyse = endTestData + 1
  val endAnalyse = beginAnalyse + analyseDataSize

  val getDataframe = dataSet(ratingAttrName, titleAttrName, reviewAttrName)

  val ngrams: List[Set[String]] = getDataframe(beginTestData, endTestData)(titleAttrName, reviewAttrName)
    .map(row => {
      (row.string(0).noPunct + " " + row.string(1).noPunct).normalize.stemmedNgram(2, 1).filter(v => (v._1.size == 2 && v._1.exists(_.length > 2) && v._1.contains("не")) || (v._1.size == 1 && v._1.exists(_.length > 2)))
    }).reduce(_ |+| _).keySet.toList

  val ngramNames = ngrams.map(ng => ng.mkString("_")).toArray
  val ngramAttributes: Array[Attribute] = ngramNames.map(ng => new NumericAttribute(ng))
  val testAttributes: Array[Attribute] = ngramAttributes ++ Array(new NumericAttribute(ratingAttrName))

  val learn = getDataframe(beginTestData, endTestData).map(testAttributes)(row => {
    val str = (row.string(1).noPunct + " " + row.string(2).noPunct).normalize
    val ngram = str.stemmedNgram(2, 1).filter(v => (v._1.size == 2 && v._1.exists(_.length > 2) && v._1.contains("не")) || (v._1.size == 1 && v._1.exists(_.length > 2)))
    ngrams.map(ng => ngram.getOrElse(ng, 0).toDouble).toArray ++ Array(row.string(0).toDouble)
  })
  val randomForestTest: RandomForest = randomForest(
    x = learn(ngramNames.toSeq: _*).unzip,
    y = learn(ratingAttrName).vector(),
    ntrees = 100
  )

  val resultStrings = new util.ArrayList[String]()
  val resultInitialResult = new util.ArrayList[Double]()

  val analyse = getDataframe(beginAnalyse, endAnalyse).map(ngramAttributes)(row => {
    val str = (row.string(1).noPunct + " " + row.string(2).noPunct).normalize
    val ngram = str.stemmedNgram(2, 1).filter(v => (v._1.size == 2 && v._1.exists(_.length > 2) && v._1.contains("не")) || (v._1.size == 1 && v._1.exists(_.length > 2)))
    resultInitialResult.add(row.string(0).toDouble)
    resultStrings.add(str)
    ngrams.map(ng => ngram.getOrElse(ng, 0).toDouble).toArray
  })

  val result = randomForestTest.predict(analyse.unzip)
  val resultZipped: mutable.Seq[(String, (Double, Double))] = resultStrings.asScala.zip(resultInitialResult.asScala.zip(result))

  val standardDeviation = Math.sqrt(resultZipped.map(t => Math.pow(t._2._1 - t._2._2, 2)).sum / resultZipped.size)
  val errorMean = Math.abs(resultZipped.map(t => t._2._1 - t._2._2).sum / resultZipped.size)

  val classifiers: mutable.Seq[(String, Double, Long)] = resultZipped.map(t => (t._1, t._2._1, if (t._2._2 < 1) {
    1
  } else if (t._2._2 > 5) {
    5
  } else {
    Math.round(t._2._2)
  }))

  val precision = (1 to 5).map(c => {
    val relevant = classifiers.filter(_._2 == c).map(_._1)
    val retrieved = classifiers.filter(_._3 == c).map(_._1)

    val precision = relevant.count(retrieved.contains(_)) / retrieved.size.toDouble
    val recall = relevant.count(retrieved.contains(_)) / relevant.size.toDouble

    (c, 2 * ((precision * recall) / (precision + recall)))
  }).toMap

  println()
  println(s"Обучающие примеры {$beginTestData, $endTestData}")
  println(s"Анализируемые строки {$beginAnalyse, $endAnalyse}")
  println(s"Стандартное отклонение: $standardDeviation")
  println(s"Результирующая ошибка: $errorMean")
  println(s"F1 scores: \n${precision.map(e => s"${e._1} -> ${e._2}\n").mkString("")}")
  println(s"F1 average: ${precision.values.sum / precision.size.toDouble}")
  println()

  implicit class MyCustomString(s: String) {
    def noPunct: String = if (s != null && !s.isEmpty) s.replaceAll("""\p{Punct}""", " ") else ""

    def stemmedNgram(maxNGramSize: Int, minFreq: Int): Map[Set[String], Int] = ngram(
      maxNGramSize,
      minFreq,
      s.split(" ").map(RussianStemmer.stem).mkString(" ")
    ).flatten.map(ng => (Set(ng.words: _*), ng.freq)).toMap
  }

}

