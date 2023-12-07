import sys

from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType

def with_column_index(sdf):
    new_schema = StructType(sdf.schema.fields + [StructField("ColumnIndex", LongType(), False),])
    return sdf.rdd.zipWithIndex().map(lambda row: row[0] + (row[1],)).toDF(schema=new_schema)

def main(spark):
    data = spark.read.json('/scratch/yx1797/nlp_data/dataset/x00*.ndjson')
    data.createOrReplaceTempView('data')
    print('Getting data...')
    df1 = data.select(posexplode(col('posts.com')).alias('index', 'text'))
    df2 = data.select(posexplode(col('posts.perspectives')).alias('index', 'label'))
    df1 = with_column_index(df1)
    df2 = with_column_index(df2)
    final = df1.join(df2, df1.ColumnIndex == df2.ColumnIndex, 'inner').drop("ColumnIndex").select('text', 'label')
    print('Getting labels...')
    # extract labels from dictionary
    test = final.select('text', 'label', get_json_object(final.label, '$.TOXICITY').alias('toxic_'), get_json_object(final.label, '$.SEVERE_TOXICITY').alias('severe_toxic_'),
                        get_json_object(final.label, '$.OBSCENE').alias('obscene_'), get_json_object(final.label, '$.INFLAMMATORY').alias('threat_'),
                        get_json_object(final.label, '$.INSULT').alias('insult_'), get_json_object(final.label, '$.PROFANITY').alias('identity_hate_'))
    # change null values to 0
    test = test.na.fill(value=0)
    # one hot encode labels based on whether label >= 0.5 or label < 0.5
    # certain columns in the original dataset do not appear in this dataset; we set their value to 0 using the floor function
    print('One-hot encoding...')
    test = test.withColumn("toxic", round(test["toxic_"]).cast('integer')).withColumn("severe_toxic", round(test["severe_toxic_"]).cast('integer'))\
        .withColumn("obscene", round(test["obscene_"]).cast('integer')).withColumn("insult", round(test["insult_"]).cast('integer'))\
        .withColumn("threat", floor(test["threat_"]).cast('integer')).withColumn("identity_hate", floor(test["identity_hate_"]).cast('integer'))
    test = test.select('text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')
    test = test.withColumn("label", array('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'))\
        .drop('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')
    test.show()
    out_dir = '/scratch/yx1797/nlp_data/preprocessed_data'
    print('Writing data...')
    test.write.json(out_dir, mode='overwrite')

# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('part2').getOrCreate()
    main(spark)