package com.sparkProject

import org.apache.spark.sql.SparkSession

object Job {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .master("local")
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    import spark.implicits._


    /********************************************************************************
      *
      *        TP 1
      *
      *        - Set environment, InteliJ, submit jobs to Spark
      *        - Load local unstructured data
      *        - Word count , Map Reduce
      ********************************************************************************/



    // ----------------- word count ------------------------

    val df_wordCount = sc.textFile("///Users/havard-macpro/Documents/Spark/spark-2.0.0-bin-hadoop2.6/README.md")
      .flatMap{case (line: String) => line.split(" ")}
      .map{case (word: String) => (word, 1)}
      .reduceByKey{case (i: Int, j: Int) => i + j}
      .toDF("word", "count")

    df_wordCount.orderBy($"count".desc).show()


    /********************************************************************************
      *
      *        TP 2 : début du projet
      *
      ********************************************************************************/

//def csv(path : ) : toto

    val df = spark
            .read
            .option("comment" , "#")
            .option("inferSchema", "true")
            .option("header" , "True")
      //      .option("sep" , "|")
            .csv("/Users/havard-macpro/Documents/Telecom paristech/cours/INF 729 - hadoop/sparks/tp_spark/cumulative.csv")

    df.show()

    println("number of columns", df.columns.length)
    //df.columns returns an Array of columns names, and arrays have a method “length” returning their length
    println("number of rows", df.count)

    df.show()

    import org.apache.spark.sql.functions._

    val columns = df.columns.slice(10, 20)
    // df.columns returns an Array. In scala arrays have a method “slice” returning a slice of the array
    df.select(columns.map(col): _*).show(50) //

  // question 3.e : afficher le shéma

    df.printSchema()

    // question 3.f : afficher le nombre d'elements

    // println("le nombre d element est :", df.groupBy("koi_disposition").count())
    df.groupBy("koi_disposition").count().show()

    // println("le nombre d element pour ", df.groupBy("koi_disposition"), "est :", df.groupBy("koi_disposition").count())


  // question 4.a

    val df2 = df.filter($"koi_disposition" === "CONFIRMED" || $"koi_disposition" === "FALSE POSITIVE" )

    // $ definit une colonne

    // si teste sur une colonne alors ===

    // || definit le ou


    df2.show()


    // question 4.b : regarder les modaklites de la colonne

    df2.groupBy("koi_eccen_err1").count().show()


    // question 4.c : comme la colonne est vide on la drop


    val df3 = df2.drop($"koi_eccen_err1")

    df3.printSchema()

    // question 4.d :
    // pour enlever une colonne :
    //        val df4 = df3.drop($"index")
    //        val df4 = df3.drop("index")
    // pour enlever plusieurs colonnes


     // val df4 = df3.drop("index", "kepid")

    // question 4.d :d.	La liste des colonnes à enlever du dataFrame avec une seule commande:

    val df4 = df3.drop("index", "kepid" ,"koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec","koi_sparprov", "koi_trans_mod","koi_datalink_dvr","koi_datalink_dvs", "koi_tce_delivname", "koi_parm_prov", "koi_limbdark_mod", "koi_fittype", "koi_disp_prov", "koi_comment", "kepoi_name", "kepler_name", "koi_vet_date", "koi_pdisposition")

    df4.printSchema()

    // question 4.e :D’autres colonnes ne contiennent qu’une seule valeur

    val useless = for(col <- df4.columns if df4.select(col).distinct().count() <= 1 ) yield col
    val df5 = df4.drop(useless: _*)

    // question 4.f : Afficher des statistiques sur les colonnes du dataFrame

    df5.describe("koi_impact", "koi_duration").show()

  // question 4.g : Certaines cellules du dataFrame ne contiennent pas de valeur. Remplacer toutes les valeurs manquantes par zéro


    val df_filled = df5.na.fill(0.0)


    // question 5 : jointure de deux dataframes

    //creation de deux dataframes

    val df_labels = df_filled.select("rowid", "koi_disposition")

    val df_features = df_filled.drop("koi_disposition")

    val df_joined = df_features.join(df_labels, usingColumn = "rowid")

    df_joined.show()

    // question 6 :

    def udf_sum = udf((col1: Double, col2: Double) => col1 + col2)


    val df_newFeatures = df_joined.withColumn("koi_ror_min", udf_sum($"koi_ror", $"koi_ror_err2")).withColumn("koi_ror_max", $"koi_ror" + $"koi_ror_err1")

    // question 7 :sauvegarder un dataframe



    df_newFeatures
      .coalesce(1) // optional : regroup all data in ONE partition, so that results are printed in ONE file
      // >>>> You should not that in general, only when the data are small enough to fit in the memory of a single machine.
      .write
      .mode("overwrite")
      .option("header", "true")
        .csv("/Users/havard-macpro/Documents/Spark/spark-2.0.0-bin-hadoop2.6/dataset")





  }




}
