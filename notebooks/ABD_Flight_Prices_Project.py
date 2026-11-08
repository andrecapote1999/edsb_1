# Databricks notebook source
# MAGIC %md
# MAGIC # ABD Project - Flight Prices
# MAGIC
# MAGIC Analysing Big Data 2026/2027 - NOVA IMS
# MAGIC
# MAGIC Group: [number] <br>
# MAGIC Members: [name, student number], [name, student number], [name, student number], [name, student number], [name, student number]

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Problem and motivation
# MAGIC
# MAGIC Websites like Skyscanner compare millions of flight prices every day. Prices change a lot depending on how early you buy,
# MAGIC the route, the airline and the day of the week. As travellers, we always ask the same questions: when should I buy,
# MAGIC which route or airline is cheaper, and is the price I'm seeing now a good deal?
# MAGIC
# MAGIC The Skyscanner API is only available for commercial partners, so we used a public dataset from Kaggle with real flight
# MAGIC prices collected from Expedia (another flight search website). The full dataset has around 80 million rows (30 GB),
# MAGIC which is too big for Databricks Free Edition, so we worked with a 1% random sample.
# MAGIC
# MAGIC **Our questions:**
# MAGIC 1. How many days in advance should we buy to get a cheaper flight?
# MAGIC 2. Which routes and airlines are the cheapest, and on which days of the week?
# MAGIC 3. Can we detect good deals as soon as new prices arrive? (streaming)
# MAGIC 4. Do cheaper airlines get more complaints from passengers? (text reviews)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Data and architecture
# MAGIC
# MAGIC **Data sources:**
# MAGIC * Flight prices (structured): `itineraries_sample.csv` - sample of the Kaggle dataset *dilwong/flightprices* (US flights, April to October 2022)
# MAGIC * Airports (structured, reference data): `airports.dat` from OpenFlights
# MAGIC * Passenger reviews (unstructured): `airline_reviews.csv` from Kaggle (free text)
# MAGIC * New prices (streaming): small JSON files that we generate to simulate new prices arriving
# MAGIC
# MAGIC **Architecture:** we followed the Medallion architecture from class:
# MAGIC * **Bronze** - raw data cleaned (data types, nulls, duplicates), saved as Delta files
# MAGIC * **Silver** - business rules (new columns and join with the airports)
# MAGIC * **Gold** - aggregated tables to answer our questions

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Set-up the environment

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create the schema for our tables and the volume for our files
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.flights COMMENT 'ABD project - flight prices';
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.flights.lakehouse COMMENT 'Files for the ABD project';

# COMMAND ----------

# Import libraries

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import datetime
import json
import random
import uuid

# COMMAND ----------

# Path variables for the folders of our Data Lake (all inside the volume)

base_path = "/Volumes/workspace/flights/lakehouse"
landing_path = base_path + "/landing"            # where we upload the source files
bronze_path = base_path + "/bronze"              # bronze layer
stream_path = base_path + "/stream_input"        # folder read by the streaming query
checkpoint_path = base_path + "/checkpoints"     # checkpoints of the streaming query

dbutils.fs.mkdirs(landing_path)
dbutils.fs.mkdirs(bronze_path)
dbutils.fs.mkdirs(stream_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Upload the 3 files to the landing folder (Catalog > workspace > flights > lakehouse > landing > Upload to this volume)
# MAGIC and check that they are there:

# COMMAND ----------

display(dbutils.fs.ls(landing_path))

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Bronze layer
# MAGIC Here we only apply data rules: correct data types, remove nulls and duplicates.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Flight prices

# COMMAND ----------

# Read the csv file (we don't use inferSchema, we change the data types ourselves later)

flights_raw = spark.read.option("header", True).csv(landing_path + "/itineraries_sample.csv")

print("Number of rows:", flights_raw.count())
flights_raw.printSchema()

# COMMAND ----------

display(flights_raw.limit(5))

# COMMAND ----------

# Check for nulls in the most important columns

flights_raw.select([count(when(isnull(c), True)).alias(c)
                    for c in ["legId", "searchDate", "flightDate", "startingAirport", "destinationAirport", "totalFare"]]).show()

# COMMAND ----------

# Drop the rows without date, airports or price

flights_noNull = flights_raw.dropna(subset=["searchDate", "flightDate", "startingAirport", "destinationAirport", "totalFare"])

# Drop duplicates (same flight, same search day and same price)
flights_noDup = flights_noNull.dropDuplicates(["legId", "searchDate", "totalFare"])

print("Rows before:", flights_raw.count())
print("Rows after:", flights_noDup.count())

# COMMAND ----------

# Change the data types (all columns came as string)

flights_bz = flights_noDup \
    .withColumn("searchDate", to_date("searchDate")) \
    .withColumn("flightDate", to_date("flightDate")) \
    .withColumn("totalFare", col("totalFare").cast("double")) \
    .withColumn("totalTravelDistance", col("totalTravelDistance").cast("int")) \
    .withColumn("seatsRemaining", col("seatsRemaining").cast("int")) \
    .withColumn("isNonStop", col("isNonStop").cast("boolean")) \
    .withColumn("isBasicEconomy", col("isBasicEconomy").cast("boolean")) \
    .withColumn("ingestion_date", current_timestamp())

flights_bz.printSchema()

# COMMAND ----------

# Save in the bronze layer as Delta, partitioned by search date
# (this way Spark only reads the days it needs)

flights_bz.write.format("delta").mode("overwrite").partitionBy("searchDate").save(bronze_path + "/flights")

# COMMAND ----------

display(dbutils.fs.ls(bronze_path + "/flights"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Airports

# COMMAND ----------

# The file has no header, so we define the schema

airports_schema = StructType([
    StructField("airport_id", IntegerType()),
    StructField("name", StringType()),
    StructField("city", StringType()),
    StructField("country", StringType()),
    StructField("iata", StringType()),
    StructField("icao", StringType()),
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()),
    StructField("altitude", IntegerType()),
    StructField("timezone", StringType()),
    StructField("dst", StringType()),
    StructField("tz", StringType()),
    StructField("type", StringType()),
    StructField("source", StringType())
])

airports_raw = spark.read.schema(airports_schema).option("nullValue", "\\N").csv(landing_path + "/airports.dat")

# Keep only airports with an IATA code (ex: JFK, LIS) and the columns we need
airports_bz = airports_raw.where(col("iata").isNotNull()) \
    .dropDuplicates(["iata"]) \
    .select("iata", "name", "city", "country")

airports_bz.write.format("delta").mode("overwrite").save(bronze_path + "/airports")
airports_bz.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Reviews

# COMMAND ----------

# Reviews are free text with commas and line breaks, so we need multiLine

reviews_raw = spark.read.option("header", True).option("multiLine", True).option("escape", '"') \
    .csv(landing_path + "/airline_reviews.csv")

reviews_raw.printSchema()

# COMMAND ----------

# Keep only the airline and the review text (change the column names if your file is different)

reviews_bz = reviews_raw.select(col("`Airline Name`").alias("airline"), col("Review").alias("review")) \
    .dropna() \
    .dropDuplicates()

reviews_bz.write.format("delta").mode("overwrite").save(bronze_path + "/reviews")
reviews_bz.show(3, truncate=80)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Silver layer
# MAGIC Here we apply the business rules: new columns that we need for the analysis and the join with the airports.

# COMMAND ----------

flights = spark.read.format("delta").load(bronze_path + "/flights")
airports = spark.read.format("delta").load(bronze_path + "/airports")

# COMMAND ----------

# New columns:
# - route (ex: JFK-LAX)
# - days between the search and the flight
# - booking window (groups of days)
# - day of the week of the flight
# - main airline (the "segments" columns have one value per flight, separated by ||)

flights_sv = flights \
    .withColumn("route", concat_ws("-", "startingAirport", "destinationAirport")) \
    .withColumn("days_in_advance", datediff("flightDate", "searchDate")) \
    .withColumn("booking_window", when(col("days_in_advance") <= 7, "1. 0-7 days")
                                 .when(col("days_in_advance") <= 14, "2. 8-14 days")
                                 .when(col("days_in_advance") <= 30, "3. 15-30 days")
                                 .when(col("days_in_advance") <= 60, "4. 31-60 days")
                                 .otherwise("5. more than 60 days")) \
    .withColumn("weekday", date_format("flightDate", "E")) \
    .withColumn("airline", split("segmentsAirlineName", "\\|\\|").getItem(0))

# COMMAND ----------

# Join with the airports to get the city names
# (airports is a small table, so we use broadcast: it is copied to all workers and the big table doesn't move)

origin = airports.select(col("iata").alias("startingAirport"), col("city").alias("origin_city"))
destination = airports.select(col("iata").alias("destinationAirport"), col("city").alias("destination_city"))

flights_sv = flights_sv \
    .join(broadcast(origin), "startingAirport", "left") \
    .join(broadcast(destination), "destinationAirport", "left") \
    .select("legId", "searchDate", "flightDate", "days_in_advance", "booking_window", "weekday",
            "route", "startingAirport", "origin_city", "destinationAirport", "destination_city",
            "airline", "isNonStop", "isBasicEconomy", "seatsRemaining", "totalTravelDistance", "totalFare")

# COMMAND ----------

# Save as a Silver table

flights_sv.write.format("delta").mode("overwrite").saveAsTable("flights.sv_flights")

display(spark.table("flights.sv_flights").limit(5))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS n_prices,
# MAGIC        COUNT(DISTINCT route) AS n_routes,
# MAGIC        COUNT(DISTINCT airline) AS n_airlines,
# MAGIC        MIN(searchDate) AS first_search,
# MAGIC        MAX(searchDate) AS last_search,
# MAGIC        ROUND(AVG(totalFare), 2) AS avg_price
# MAGIC FROM flights.sv_flights

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Gold layer
# MAGIC One table for each question.

# COMMAND ----------

sv = spark.table("flights.sv_flights")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.1 Question 1: how many days in advance should we buy?

# COMMAND ----------

gd_booking_window = sv.groupBy("booking_window") \
    .agg(count("*").alias("n_prices"),
         round(avg("totalFare"), 2).alias("avg_price"),
         round(min("totalFare"), 2).alias("min_price"))

gd_booking_window.write.format("delta").mode("overwrite").saveAsTable("flights.gd_booking_window")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Bar chart: booking_window vs avg_price
# MAGIC SELECT * FROM flights.gd_booking_window ORDER BY booking_window

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2 Question 2: cheapest routes, airlines and days

# COMMAND ----------

# Average price per route and ranking of the cheapest routes from each airport (window function)

window_origin = Window.partitionBy("startingAirport").orderBy("avg_price")

gd_routes = sv.groupBy("route", "startingAirport", "destinationAirport", "destination_city") \
    .agg(count("*").alias("n_prices"),
         round(avg("totalFare"), 2).alias("avg_price"),
         round(avg(col("isNonStop").cast("int")) * 100, 1).alias("pct_nonstop")) \
    .where(col("n_prices") >= 30) \
    .withColumn("rank", dense_rank().over(window_origin))

gd_routes.write.format("delta").mode("overwrite").saveAsTable("flights.gd_routes")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Top 3 cheapest destinations from each airport
# MAGIC SELECT startingAirport, rank, destinationAirport, destination_city, avg_price, pct_nonstop
# MAGIC FROM flights.gd_routes
# MAGIC WHERE rank <= 3
# MAGIC ORDER BY startingAirport, rank

# COMMAND ----------

# Average price per airline and day of the week (pivot)

gd_airline_weekday = sv.groupBy("airline") \
    .pivot("weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]) \
    .agg(round(avg("totalFare"), 2))

gd_airline_weekday.write.format("delta").mode("overwrite").saveAsTable("flights.gd_airline_weekday")
display(spark.table("flights.gd_airline_weekday"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.3 Average price per route (used later by the streaming)

# COMMAND ----------

gd_route_avg = sv.groupBy("route") \
    .agg(count("*").alias("n_prices"), round(avg("totalFare"), 2).alias("avg_price")) \
    .where(col("n_prices") >= 30)

gd_route_avg.write.format("delta").mode("overwrite").saveAsTable("flights.gd_route_avg")
gd_route_avg.orderBy(desc("n_prices")).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Reviews (Question 4)
# MAGIC We used the Word Count from class to see the most common words in the reviews,
# MAGIC and then we checked how many reviews talk about problems (delays, lost bags, refunds...).

# COMMAND ----------

reviews = spark.read.format("delta").load(bronze_path + "/reviews")

# Clean the text (lower case, remove punctuation), split in words and count
stop_words = ["the", "and", "was", "were", "for", "that", "with", "this", "they", "but", "not", "had",
              "have", "from", "our", "you", "are", "flight", "airline", "very", "there", "would", "their",
              "when", "all", "after", "been", "one", "out", "get", "got", "did", "has", "verified", "trip"]

words = reviews.select(explode(split(regexp_replace(lower("review"), "[^a-z ]", " "), " ")).alias("word")) \
    .where(length("word") > 2) \
    .where(~col("word").isin(stop_words))

words.groupBy("word").count().orderBy(desc("count")).show(20)

# COMMAND ----------

# Mark the reviews that mention a problem

problems = "delay|delayed|cancel|cancelled|lost|refund|rude|late|dirty|worst|terrible"

reviews_problems = reviews \
    .withColumn("has_problem", lower("review").rlike(problems)) \
    .withColumn("airline_key", lower(split("airline", " ").getItem(0)))

# The airline names are a bit different in the two datasets (ex: "Delta" and "Delta Air Lines"),
# so we join them by the first word of the name

complaints = reviews_problems.groupBy("airline_key") \
    .agg(count("*").alias("n_reviews"),
         round(avg(col("has_problem").cast("int")) * 100, 1).alias("pct_complaints"))

prices = sv.withColumn("airline_key", lower(split("airline", " ").getItem(0))) \
    .groupBy("airline_key") \
    .agg(first("airline").alias("airline"), round(avg("totalFare"), 2).alias("avg_price"))

gd_complaints = prices.join(complaints, "airline_key")

gd_complaints.write.format("delta").mode("overwrite").saveAsTable("flights.gd_complaints")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT airline, avg_price, n_reviews, pct_complaints
# MAGIC FROM flights.gd_complaints
# MAGIC ORDER BY avg_price

# COMMAND ----------

# MAGIC %md
# MAGIC # 8. Streaming (Question 3)
# MAGIC We simulate new prices arriving as JSON files in a folder (like the example from class 8).
# MAGIC The streaming query reads the new files, compares each price with the average price of the route (Gold table)
# MAGIC and marks it as a deal if it is 30% or more below the average.
# MAGIC
# MAGIC Note: in Databricks Free Edition we have to use `trigger(availableNow=True)`, so the query processes the new files and then stops.

# COMMAND ----------

# Clean the folders from previous runs

dbutils.fs.rm(stream_path, True)
dbutils.fs.rm(checkpoint_path, True)
dbutils.fs.mkdirs(stream_path)
spark.sql("DROP TABLE IF EXISTS flights.st_prices")

# COMMAND ----------

# Function to create a file with new prices
# We use real routes and average prices from the Gold table, and some prices are much lower (deals)

routes = spark.table("flights.gd_route_avg").orderBy(desc("n_prices")).limit(30).collect()

def new_prices_file(n=20):
    rows = []
    for i in range(n):
        r = random.choice(routes)
        if random.random() < 0.2:
            price = r["avg_price"] * random.uniform(0.4, 0.7)    # deal
        else:
            price = r["avg_price"] * random.uniform(0.8, 1.3)    # normal price
        rows.append(json.dumps({
            "id": str(uuid.uuid4()),
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "route": r["route"],
            "price": float("%.2f" % price)
        }))
    file_name = stream_path + "/prices_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    dbutils.fs.put(file_name, "\n".join(rows), True)
    print("New file:", file_name)

# COMMAND ----------

# 3 files arrive

new_prices_file()
new_prices_file()
new_prices_file()

# COMMAND ----------

# Read the stream (for files we need to give the schema)

prices_schema = StructType([
    StructField("id", StringType()),
    StructField("time", StringType()),
    StructField("route", StringType()),
    StructField("price", DoubleType())
])

prices_stream = spark.readStream.format("json").schema(prices_schema).load(stream_path)

# COMMAND ----------

# Transformation: join with the average price of each route and check if it is a deal

route_avg = spark.table("flights.gd_route_avg").select("route", "avg_price")

deals_stream = prices_stream \
    .withColumn("time", to_timestamp("time")) \
    .join(route_avg, "route", "left") \
    .withColumn("pct_vs_avg", round((col("price") / col("avg_price") - 1) * 100, 1)) \
    .withColumn("is_deal", col("price") <= col("avg_price") * 0.7)

# COMMAND ----------

# Write the stream to a Delta table

query = deals_stream.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", checkpoint_path) \
    .trigger(availableNow=True) \
    .toTable("flights.st_prices")

query.awaitTermination()

print("Rows in the table:", spark.table("flights.st_prices").count())

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The deals found
# MAGIC SELECT time, route, price, avg_price, pct_vs_avg
# MAGIC FROM flights.st_prices
# MAGIC WHERE is_deal
# MAGIC ORDER BY pct_vs_avg

# COMMAND ----------

# MAGIC %md
# MAGIC # 9. Delta Lake: table with the best price per route (MERGE and time travel)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE flights.gd_best_price (
# MAGIC   route STRING,
# MAGIC   best_price DOUBLE,
# MAGIC   time TIMESTAMP
# MAGIC ) USING DELTA

# COMMAND ----------

# Get the lowest price of each route from the streaming table and create a temporary view to use in the MERGE

spark.table("flights.st_prices").groupBy("route") \
    .agg(min("price").alias("best_price"), max("time").alias("time")) \
    .createOrReplaceTempView("new_best_prices")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Update the route if the new price is lower, insert the route if it is new
# MAGIC MERGE INTO flights.gd_best_price AS a
# MAGIC USING new_best_prices AS b
# MAGIC ON a.route = b.route
# MAGIC WHEN MATCHED AND b.best_price < a.best_price THEN
# MAGIC   UPDATE SET a.best_price = b.best_price, a.time = b.time
# MAGIC WHEN NOT MATCHED THEN
# MAGIC   INSERT (route, best_price, time) VALUES (b.route, b.best_price, b.time)

# COMMAND ----------

# MAGIC %md
# MAGIC Now 3 more files arrive. We run the same streaming query again and only the new files are read (because of the checkpoint).

# COMMAND ----------

new_prices_file()
new_prices_file()
new_prices_file()

query = deals_stream.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", checkpoint_path) \
    .trigger(availableNow=True) \
    .toTable("flights.st_prices")

query.awaitTermination()

print("Rows in the table:", spark.table("flights.st_prices").count())

# COMMAND ----------

# Update the view with the new prices and run the MERGE again

spark.table("flights.st_prices").groupBy("route") \
    .agg(min("price").alias("best_price"), max("time").alias("time")) \
    .createOrReplaceTempView("new_best_prices")

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO flights.gd_best_price AS a
# MAGIC USING new_best_prices AS b
# MAGIC ON a.route = b.route
# MAGIC WHEN MATCHED AND b.best_price < a.best_price THEN
# MAGIC   UPDATE SET a.best_price = b.best_price, a.time = b.time
# MAGIC WHEN NOT MATCHED THEN
# MAGIC   INSERT (route, best_price, time) VALUES (b.route, b.best_price, b.time)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- History of the table (one version for each change)
# MAGIC DESCRIBE HISTORY flights.gd_best_price

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Time travel: compare the first MERGE (version 1) with the current version
# MAGIC SELECT a.route, b.best_price AS price_version_1, a.best_price AS price_now
# MAGIC FROM flights.gd_best_price a
# MAGIC LEFT JOIN flights.gd_best_price VERSION AS OF 1 b ON a.route = b.route
# MAGIC ORDER BY a.route

# COMMAND ----------

# MAGIC %sql
# MAGIC -- All the tables we created
# MAGIC SHOW TABLES IN flights

# COMMAND ----------

# MAGIC %md
# MAGIC # 10. Conclusions
# MAGIC
# MAGIC *(write after running the notebook, with the real numbers)*
# MAGIC
# MAGIC 1. **When to buy:** ...
# MAGIC 2. **Cheapest routes, airlines and days:** ...
# MAGIC 3. **Deals with streaming:** ...
# MAGIC 4. **Complaints vs price:** ...
# MAGIC
# MAGIC **Limitations:** we used a 1% sample because of the limits of Databricks Free Edition (the code would be the same for the full dataset),
# MAGIC the streaming data is simulated, and the data is only from US flights in 2022.
