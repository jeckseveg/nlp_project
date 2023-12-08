import sys
import glob
import argparse
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import StructType, StructField, LongType


def to_null(c):
    return when(~(col(c).isNull() | isnan(col(c)) | (trim(col(c)) == "")), col(c))


def with_column_index(sdf):
    new_schema = StructType(sdf.schema.fields + [StructField("ColumnIndex", LongType(), False),])
    return sdf.rdd.zipWithIndex().map(lambda row: row[0] + (row[1],)).toDF(schema=new_schema)


def main(spark, args):
    prefix = str(args.folder)
    train_path = '/scratch/yx1797/nlp_data/preprocessed_data/train/'+prefix
    val_path = '/scratch/yx1797/nlp_data/preprocessed_data/val/'+prefix
    print('Getting data...')
    data = spark.read.json('/scratch/yx1797/nlp_data/preprocessed_data/preprocessed/'+prefix+'*.json')
    data.createOrReplaceTempView('data')
    # remove null/empty values
    print('Removing null or empty values...')
    data = data.select([to_null('text').alias('text'), col('label')]).na.drop()
    data = data.orderBy('text').orderBy(rand(seed=42))
    # now that df is ordered randomly, generate indicies for random order
    data = with_column_index(data)
    window = Window.orderBy('ColumnIndex')
    train = data.select('*', percent_rank().over(window).alias('rank')).filter(col('rank') < 0.8).drop('rank').drop('ColumnIndex')
    val = data.select('*', percent_rank().over(window).alias('rank')).filter(col('rank') >= 0.8).drop('rank').drop('ColumnIndex')
    data.show()
    train.show()
    val.show()
    print('Writing data to folder '+prefix+'...')
    train.coalesce(1).write.csv(train_path, mode='append', header=True)
    val.coalesce(1).write.csv(val_path, mode='append', header=True)

# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('part2').getOrCreate()
    parser = argparse.ArgumentParser(description="prefix to process and save out")
    parser.add_argument("--folder", type=str, default='part-00000', help="prefix of the json file")
    args = parser.parse_args()
    main(spark, args)