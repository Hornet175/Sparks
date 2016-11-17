package com.sparkProject

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator}



/**
  * Created by havard-macpro on 25/10/2016.
  */


object JobML {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .master("local")
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    import spark.implicits._


    val df = spark.read.parquet("/Users/havard-macpro/Documents/Telecom_paristech/cours/INF_729_-_hadoop/sparks/tp_spark/cleanedDataFrame.parquet")


    val df2 = df.drop("rowid")

    val features = df2.columns.filter(x => x != "koi_disposition")


    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")

    val df3 = assembler.transform(df2)



    val df4 = df3.select("features", "koi_disposition")

    // normalisation //


    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(df4)

    // Normalize each feature to have unit standard deviation.
    val df5 = scalerModel.transform(df4)

    //conversion des chaines de caractères //

    val indexer = new StringIndexer()
      .setInputCol("koi_disposition")
      .setOutputCol("label")

    val df6 = indexer.fit(df5).transform(df5)

    df6.show()

    // split des données //

    val splits = df6.randomSplit(Array(0.9, 0.1))
    val (trainingData, testData) = (splits(0), splits(1))

    trainingData.show()
    testData.show()

    trainingData.cache()
    testData.cache()

    // Logistic Regression //

    val lr = new LogisticRegression()
      .setElasticNetParam(1.0)
      .setLabelCol("label")
      .setStandardization(true)
      .setFitIntercept(true)
      .setTol(1.0e-5)
      .setMaxIter(300)

    // calcul des hyper parametre avec grid search//

    val array = -6.0 to (0.0, 0.5) toArray
    val arrayLog = array.map(x => math.pow(10,x))

    print(arrayLog.deep.mkString(" | ") + "\n\n")

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, arrayLog)
      .build()



  // split du modele calculé//

    val evaluation = new BinaryClassificationEvaluator().setLabelCol("label")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(evaluation)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)


    val lr2 = trainValidationSplit.fit(trainingData)

    // calcul des predictions//

    val predicts = lr2.transform(testData).select("features", "label", "prediction")

    predicts.groupBy("label", "prediction").count.show()
    evaluation.setRawPredictionCol("prediction")
    println("Model accuracy : "  + evaluation.evaluate(predicts).toString())

    // sauvegarde du modèle //

    sc.parallelize(Seq(lr2), 1).saveAsObjectFile("/Users/havard-macpro/Documents/Telecom_paristech/cours/INF_729_-_hadoop/sparks/tp_spark/TP_jmh.model")
  }
}
