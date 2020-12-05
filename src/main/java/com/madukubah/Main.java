package com.madukubah;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.SparkSession;

import com.madukubah.classification.Classification;

public class Main {

	public static void main(String[] args) {
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		
		Boolean verbose = false;
		
		Options options = new Options();
		options.addOption("verbose", true, "verbose");

		CommandLineParser clparser = new PosixParser();
		CommandLine cm;
		try	
		{
			cm = clparser.parse(options, args);
			SparkSession spark = SparkSession.builder().appName("analitic").master("local[*]").getOrCreate();
			if (cm.hasOption("verbose")) {
				verbose = true;
			}
			Classification postClassification = new Classification( spark );
			postClassification.setVerbose(verbose).doClassification("temp_targets");
			
			spark.close();
			
		} catch (ParseException e) {
			e.printStackTrace();
		}
	}

}
