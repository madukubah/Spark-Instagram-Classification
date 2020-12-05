package com.madukubah.classification;

import static org.apache.spark.sql.functions.count;
import static org.apache.spark.sql.functions.monotonically_increasing_id;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import com.madukubah.config.Config;

import scala.Tuple2;


public class Classification {
	private boolean verbose = false;
	private SparkSession ssc; 
	
	public Classification( SparkSession ssc ) 
	{
		this.ssc = ssc;
	}
	/**
	 * Set the verbose mode.
	 * @param verbose
	 * @return this object
	 */
	public Classification setVerbose(boolean verbose) {
		this.verbose = verbose;
		return this;
	}

	@SuppressWarnings("unchecked")
	public void doClassification( String table ) {
		
		Properties connectionProperties = new Properties();
		connectionProperties.put("user", Config.DB_USERNAME );
		connectionProperties.put("password", Config.DB_USERPASSWORD);
		
		Dataset<Row> target = this.ssc.read()
				  .jdbc(Config.DB_NAME, table, connectionProperties);
		Dataset<Row> posts = this.ssc.read()
				  .jdbc(Config.DB_NAME, "posts", connectionProperties);
		Dataset<Row> baseClass = this.ssc.read()
				  .jdbc(Config.DB_NAME, "base_classes", connectionProperties);
		
		if (verbose) {
			System.out.println("load target, posts, baseClass");
		}
		
		target = target.select("username").withColumnRenamed("username", "username_x");
		target = target
				.join(posts, target.col("username_x").equalTo(posts.col("username")), "inner")
				;
		target = target.select("username", "desc_image", "source_image");
		
		Dataset<Row> baseClassWords = baseClass.select("values");
		baseClassWords = createWordCount( baseClassWords );

		target = target.withColumnRenamed("desc_image", "text");
		baseClass = baseClass.withColumnRenamed("values", "text");
		baseClassWords = baseClassWords.withColumnRenamed("value", "text");
		
		Tokenizer tokenizer = new Tokenizer()
				.setInputCol("text")
				.setOutputCol("tokens");
		baseClassWords = tokenizer.transform(baseClassWords);
		target = tokenizer.transform(target);
		baseClass = tokenizer.transform(baseClass);
		
		CountVectorizer countVectorizer = new CountVectorizer()
				.setInputCol("tokens")
				.setOutputCol("features")
				;
		CountVectorizerModel countVectorizerModel = countVectorizer.fit(baseClassWords);
		target = countVectorizerModel.transform(target);
		baseClass = countVectorizerModel.transform(baseClass);
		baseClassWords = countVectorizerModel.transform(baseClassWords);
		
		CosineEstimator cosineEstimator = new CosineEstimator();
		CosineModel cosineModel = cosineEstimator.fit(baseClass);
		cosineModel.setVerbose(verbose);
		
		Dataset<Row> predictions =  cosineModel.transform(target);
		
		Dataset<Row> usernames = predictions.select("username", "prediction");
		
		JavaRDD<Row> rowRdd = (JavaRDD<Row>) usernames.toJavaRDD();
		JavaPairRDD<String, Integer> tes =  rowRdd.mapToPair( row -> {
			return new Tuple2<String, Integer>(row.getString(0), (int) row.getLong(1));
		});
		JavaPairRDD<String, Iterable<Integer>> tes2  = tes.groupByKey();
		JavaPairRDD<String, Integer> accountAnalysisRdd= tes2.mapToPair( val -> {
			Iterable<Integer> list =  val._2;
			Map<Integer,Integer> map=new HashMap<Integer,Integer>();
			for( Integer i : list )
			{
				if( map.get(i) != null )
					map.put(i, map.get(i) + 1);
				else
					map.put(i, 1 );	
			}
			int max = 0;
			int maxKey = -1;
			for( Map.Entry m:map.entrySet() )
			{
				if( map.get( m.getKey() ) > max )
				{
					max = map.get( m.getKey() );
					maxKey = (int) m.getKey();
				}
			}
			return new Tuple2(val._1, maxKey );
		} );
		
		JavaRDD<Row> accountAnalysisRow = accountAnalysisRdd.map(val->{
			return RowFactory.create(val._1, val._2 );
		});
		StructType schema = new StructType(new StructField[]{
				new StructField("username", DataTypes.StringType, false, Metadata.empty()),
				new StructField("code", DataTypes.IntegerType, false, Metadata.empty())
			});

		Dataset<Row> accountAnalysis = this.ssc.createDataFrame(accountAnalysisRow, schema); 
		
		accountAnalysis
			.write()
			.mode(SaveMode.Overwrite)
			  .format("jdbc")
			  .option("url", Config.DB_NAME)
			  .option("dbtable", "account_analysis")
			  .option("user", Config.DB_USERNAME)
			  .option("password", Config.DB_USERPASSWORD)
			  .save();
		
		baseClass = baseClass.select( "code", "class");
		predictions = predictions
				.join(baseClass, predictions.col("prediction").equalTo(baseClass.col("code")), "left")
				;
		
		predictions
			.select("username", "source_image", "text", "prediction", "value", "class")
			.write()
			.mode(SaveMode.Overwrite)
			  .format("jdbc")
			  .option("url", Config.DB_NAME)
			  .option("dbtable", "account_tendency")
			  .option("user", Config.DB_USERNAME)
			  .option("password", Config.DB_USERPASSWORD)
			  .save();
	}
	
	private Dataset<Row> createWordCount( Dataset<Row> rowStrings ) 
	{
		if (verbose) {
			System.out.println("createWordCount");
		}
		
		Dataset<String> dfword = rowStrings.flatMap(
				( FlatMapFunction<Row, String>  ) s -> {
					String sentence = s.mkString();
					sentence = sentence.replaceAll("[^a-zA-Z\\s]", "").toLowerCase().trim();
					return Arrays.asList(  sentence.split(" ") ).iterator();
				}
				 ,Encoders.STRING() );
				
		dfword = dfword
				.map( ( MapFunction<String, String> ) word -> word.trim(), 
						Encoders.STRING())
				.filter("LENGTH( TRIM( value  )) > 0");
				
		Dataset<Row> dfwordCount = dfword
				.groupBy("value")
				.agg(count("value").as("count"))
				;
		
		dfwordCount = dfwordCount.withColumn( "id", monotonically_increasing_id() );
		return dfwordCount ;
	}
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		Logger.getLogger("org.apache").warn("pos clustering");
		SparkSession spark = SparkSession.builder().appName("analitic").master("local[*]").getOrCreate();
		
		Classification postClassification = new Classification( spark );
		postClassification
//			.setVerbose(true)
			.doClassification("temp_targets");
		spark.close();
	}
}
