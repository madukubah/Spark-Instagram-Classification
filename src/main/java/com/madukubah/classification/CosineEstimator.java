package com.madukubah.classification;

import java.util.UUID;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;

public class CosineEstimator extends Estimator<CosineModel> {

	private static final long serialVersionUID = -6366026106299743238L;
	private String uid = null;
	@Override
	public String uid() {
		if (uid == null) {
			String ruid = UUID.randomUUID().toString();
			int n = ruid.length();
			uid = "cmm" + "_" + ruid.substring(n-12, n);
		}
		return uid;
	}

	@Override
	public Estimator<CosineModel> copy(ParamMap arg0) {
		return defaultCopy(arg0);
	}

	@Override
	public CosineModel fit(Dataset<?> rows) {
		JavaRDD<Row> rowRdd = (JavaRDD<Row>) rows.toJavaRDD();
		JavaRDD<IndexedRow> ir = rowRdd.map( row -> {
			int i = row.fieldIndex("features");
			Vector features = row.getAs( i );
			i = row.fieldIndex("code");
			long id = (long )row.getInt(i);
			return new IndexedRow(id, org.apache.spark.mllib.linalg.Vectors.fromML(features) );
		});
		// TODO Auto-generated method stub
		return new CosineModel( ir.collect() );
	}

	@Override
	public StructType transformSchema(StructType schema) {
		return schema;
	}

}
