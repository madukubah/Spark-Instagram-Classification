package com.madukubah.classification;

import java.util.UUID;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.SchemaUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class SimpleIndexerModel extends Model<SimpleIndexerModel> {

	@Override
	public String uid() {
		String ruid = UUID.randomUUID().toString();
		int n = ruid.length();
		return "SimpleIndexerModel" + "_" + ruid.substring(n-12, n);
	}

	@Override
	public SimpleIndexerModel copy(ParamMap arg0) {
		return defaultCopy(arg0);
	}

	@Override
	public Dataset<Row> transform(Dataset<?> arg0) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public StructType transformSchema(StructType arg0) {
		
//		int idx = arg0.fieldIndex("text");
//		StructField field = arg0.fields().clone()[idx];
		return SchemaUtils.appendColumn(arg0, new StructField("prediction", DataTypes.IntegerType, false, null ));
	}

}
