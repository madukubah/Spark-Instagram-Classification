package com.madukubah.classification;

import java.util.List;
import java.util.UUID;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.Model;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.SchemaUtils;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

public class CosineModel extends Model<CosineModel> {
	
	private List<IndexedRow> baseClass = null;
	private boolean verbose = false;
	
	public CosineModel setVerbose(boolean verbose) {
		this.verbose = verbose;
		return this;
	}
	
	public CosineModel(List<IndexedRow> baseClass ) {
		this.baseClass = baseClass;
	}
	
	@Override
	public String uid() {
		String ruid = UUID.randomUUID().toString();
		int n = ruid.length();
		return "CosineModel" + "_" + ruid.substring(n-12, n);
	}

	@Override
	public CosineModel copy(ParamMap arg0) {
		return defaultCopy(arg0);
	}

	@Override
	public Dataset<Row> transform(Dataset<?> dataset) {
		JavaRDD<Row> output = (JavaRDD<Row>) dataset.javaRDD();
		JavaRDD<Row> output2 = output.map(new ComputeFunction());
		
		 StructType schema = new StructType(new StructField[]{
			new StructField("username", DataTypes.StringType, false, Metadata.empty()),
			new StructField("source_image", DataTypes.StringType, false, Metadata.empty()),
			new StructField("text", DataTypes.StringType, false, Metadata.empty()),
			new StructField("features", new org.apache.spark.mllib.linalg.VectorUDT() , false, Metadata.empty()),
			new StructField("prediction", DataTypes.LongType, false, Metadata.empty()),
			new StructField("value", DataTypes.DoubleType, false, Metadata.empty())
		});
		return dataset.sqlContext().createDataFrame(output2, schema);
//		return null;
	}
	
	class ComputeFunction implements Function<Row, Row> {

		@Override
		public Row call(Row row) throws Exception {
			int i = row.fieldIndex("features");
			if (verbose) {
				System.out.println( row );
			}
			Vector features = org.apache.spark.mllib.linalg.Vectors.fromML( row.getAs( i ) );
			Tuple2<Long, Double> nearest = compute( features );
			return RowFactory.create(row.getString(0), row.getString(2), row.getString(1),  features , nearest._1, nearest._2 );
		}
		
		private Tuple2<Long, Double> compute(Vector v) {
			Long id = (long) -1;
			double dist = -2;
			for( IndexedRow s : baseClass ) {
				Vector y = (Vector) s.vector();
				double dot = BLAS.dot(v, y);
				double a = ( dot / ( Vectors.norm(v, 2) * Vectors.norm( y , 2) ) ) ;
				if (verbose) {
					System.out.println(s.index());
					System.out.println(v);
					System.out.println(y);
					
					System.out.println(dot);
					System.out.println(Vectors.norm(v, 2));
					System.out.println(Vectors.norm(y, 2));
					
					System.out.println(a);
					System.out.println("\n");
				}
				
				if( a > dist )
				{
					id = s.index();
					dist = a;
				}
			}
			if (verbose) {
				System.out.println("\n");
				System.out.println("\n");
			}
			return new Tuple2( id, dist );
		}
	}

	@Override
	public StructType transformSchema(StructType arg0) {

		return SchemaUtils.appendColumn(arg0, new StructField("prediction", DataTypes.IntegerType, false,  Metadata.empty() ));
	}

}
