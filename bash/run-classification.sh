#!/bin/sh
spark-submit \
--master local \
--deploy-mode client \
--executor-memory 1g \
--name classification \
--conf "spark.app.id=analitic" \
--conf spark.driver.extraClassPath=/home/madukubah/eclipse-workspace/analitic/jars/postgresql-42.2.12.jar \
/home/madukubah/eclipse-workspace/classification/target/classification-0.0.1-SNAPSHOT.jar --verbose=1

