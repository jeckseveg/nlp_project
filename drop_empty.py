import sys
import glob
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType


def to_null(c):
    return when(~(col(c).isNull() | isnan(col(c)) | (trim(col(c)) == "")), col(c))


def main(spark):
    path = '/scratch/yx1797/nlp_data/preprocessed_data/preprocessed/*.json'
    files = glob.glob(path)
    print(files)
    for file in files:
        print('Getting data...')
        data = spark.read.json(file)
        data.createOrReplaceTempView('data')
        print('Removing null or empty values...')
        data = data.select([to_null(c).alias(c) for c in data.columns]).na.drop()
        data.show()
        print('Writing data...')
        data.write.json(file, mode='overwrite')

# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('part2').getOrCreate()
    main(spark)