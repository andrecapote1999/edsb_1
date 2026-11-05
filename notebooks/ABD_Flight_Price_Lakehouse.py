# Databricks notebook source
# MAGIC %md
# MAGIC # ✈️ Flight Price Intelligence
# MAGIC ## A Skyscanner-style price Lakehouse with Apache Spark, Delta Lake and Structured Streaming
# MAGIC
# MAGIC **Course:** Analysing Big Data (ABD) 2026/2027 — NOVA IMS
# MAGIC **Group:** _[Group number]_
# MAGIC **Members:** _[Name 1 (student nº)], [Name 2], [Name 3], [Name 4], [Name 5], [Name 6]_
# MAGIC **Platform:** Databricks Free Edition (serverless compute) · PySpark · Spark SQL · Delta Lake
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Table of contents
# MAGIC 1. Problem definition and motivation
# MAGIC 2. Data sources and solution architecture
# MAGIC 3. Environment set-up
# MAGIC 4. Bronze layer: raw data ingestion and data-quality rules
# MAGIC 5. Silver layer: business rules and conformed data
# MAGIC 6. Gold layer: business aggregations (structured data)
# MAGIC 7. Unstructured data: passenger reviews (text analytics)
# MAGIC 8. Streaming data: real-time price alerts
# MAGIC 9. Delta Lake: "best price board" with MERGE and time travel
# MAGIC 10. Execution summary
# MAGIC 11. Conclusions

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Problem definition and motivation
# MAGIC
# MAGIC Flight metasearch engines such as **Skyscanner** compare millions of fares from airlines and online travel agencies every day.
# MAGIC Prices change constantly with demand, the number of days before departure, the day of the week, the route and the airline.
# MAGIC For a traveller the questions are always the same: *when should I buy, where is it cheaper, and is this price a good deal right now?*
# MAGIC
# MAGIC This is a typical **Big Data** problem:
# MAGIC
# MAGIC | V | In this project |
# MAGIC |---|---|
# MAGIC | **Volume** | The full source dataset has ~80 million fare quotes (~30 GB) |
# MAGIC | **Velocity** | New price quotes arrive continuously and must be evaluated in near real time |
# MAGIC | **Variety** | Structured fares (CSV), reference data (airports), unstructured passenger reviews (free text) and streaming quotes (JSON) |
# MAGIC
# MAGIC ### Business questions
# MAGIC | # | Question | Data type | Section |
# MAGIC |---|---|---|---|
# MAGIC | **Q1** | How does buying in advance affect the fare? What is the best booking window? | Structured | 6.1 |
# MAGIC | **Q2** | Which routes and airlines are the cheapest, and on which weekdays? | Structured | 6.2 / 6.3 |
# MAGIC | **Q3** | How do prices evolve over time on the busiest routes? | Structured | 6.4 |
# MAGIC | **Q4** | Can we detect, in near real time, quotes that are significantly below the historical price (a "price alert")? | Streaming | 8 / 9 |
# MAGIC | **Q5** | What do passengers complain about, and are cheaper airlines the ones with more complaints? | Unstructured | 7 |
# MAGIC
# MAGIC > **Note on the data source:** Skyscanner's Travel API is only available to approved commercial partners.
# MAGIC > We therefore use a public dataset of real flight fares collected from **Expedia** (another travel search engine).
# MAGIC > The data has the same nature as Skyscanner's (one row per fare quote found in a search).

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Data sources and solution architecture
# MAGIC
# MAGIC ### Data sources
# MAGIC | Source | Format | Type | Content |
# MAGIC |---|---|---|---|
# MAGIC | [Flight Prices – Expedia (Kaggle, dilwong)](https://www.kaggle.com/datasets/dilwong/flightprices) | CSV | Structured | One-way fare quotes searched between Apr and Oct 2022 for 16 major US airports (search date, flight date, route, fare, airline, stops, duration, distance…) |
# MAGIC | [OpenFlights airports](https://openflights.org/data) | CSV (`airports.dat`) | Structured (reference / dimension) | Airport IATA code, name, city, country, coordinates |
# MAGIC | Airline passenger reviews (Kaggle, Skytrax-based dataset) | CSV with free text | **Unstructured** | Review text written by passengers per airline |
# MAGIC | Simulated price quotes | JSON files arriving in a folder | **Streaming** | New fare quotes produced by "partners", as in the ABD08 streaming class |
# MAGIC
# MAGIC **Volume decision:** the full fare dataset (~30 GB) exceeds what Databricks Free Edition can handle comfortably.
# MAGIC We use a **random sample** of the file (taken from every chunk, so all routes and all months are kept).
# MAGIC The code does not depend on the sample size: on a paid cluster, the same notebook would run on the full file.
# MAGIC
# MAGIC ### Architecture: Medallion Lakehouse
# MAGIC ```
# MAGIC  LANDING (Volume)            BRONZE (Delta files)        SILVER (Delta tables)              GOLD (Delta tables)
# MAGIC  ───────────────────         ────────────────────        ─────────────────────              ───────────────────
# MAGIC  itineraries_sample.csv ──►  bz_itineraries  ─────┐
# MAGIC                              (data rules)         ├──►  sv_flights  ──────────────────►   gd_route_advance_price   (Q1)
# MAGIC  airports.dat ───────────►  bz_airports  ────────┘     sv_flight_segments (explode)       gd_route_ranking         (Q2)
# MAGIC                                                         (business rules + joins)           gd_airline_weekday_fare  (Q2)
# MAGIC                                                                                            gd_daily_price_trend     (Q3)
# MAGIC                                                                                            gd_route_price_baseline  (Q4)
# MAGIC  airline_reviews.csv ────►  bz_reviews  ──────────────► sv_review_words ────────────────►   gd_airline_top_words     (Q5)
# MAGIC                                                                                            gd_airline_price_vs_complaints (Q5)
# MAGIC
# MAGIC  stream_input/*.json ──► Structured Streaming ──► st_price_quotes (+ join with baseline) ──► gd_best_price_board (MERGE)  (Q4)
# MAGIC                                               └─► st_deals_by_route (streaming aggregation)
# MAGIC ```
# MAGIC * **Bronze:** raw data, organised by source system, with *data* rules applied (types, nulls, duplicates).
# MAGIC * **Silver:** clean and conformed history of business facts, with *business* rules applied (new attributes, joins with reference data).
# MAGIC * **Gold:** aggregated tables built for specific business questions, ready to be queried with SQL.

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Environment set-up

# COMMAND ----------

# Import libraries
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import builtins as py   # Python built-ins (max, round, sum) are shadowed by pyspark.sql.functions
import datetime
import json
import random
import uuid

random.seed(42)

# COMMAND ----------

#batatas

# COMMAND ----------

# ---------------------------------------------------------------
# Project configuration (the only cell to edit if names change)
# ---------------------------------------------------------------
CATALOG = "workspace"     # default catalog in Databricks Free Edition
SCHEMA  = "flights"       # database/schema for all project tables
VOLUME  = "lakehouse"     # Unity Catalog Volume for the Data Lake files

BASE_PATH   = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
LANDING     = f"{BASE_PATH}/landing"          # source files uploaded by the group
BRONZE      = f"{BASE_PATH}/bronze"           # Bronze layer (Delta files)
STREAM_IN   = f"{BASE_PATH}/stream_input"     # folder monitored by Structured Streaming
CHECKPOINTS = f"{BASE_PATH}/checkpoints"      # streaming checkpoint locations

# Source files (upload them to the LANDING folder, see next cells)
FILE_ITINERARIES = f"{LANDING}/itineraries_sample.csv"
FILE_AIRPORTS    = f"{LANDING}/airports.dat"
FILE_REVIEWS     = f"{LANDING}/airline_reviews.csv"

# Column names in the reviews file (different Kaggle review datasets use different names:
# check the printSchema() output in section 4.3 and adjust here if needed)
REVIEW_AIRLINE_COL = "Airline Name"
REVIEW_TEXT_COL    = "Review"

# Business parameters
DEAL_THRESHOLD   = 0.30   # a quote is a "deal" if it is at least 30% below the route average
MIN_QUOTES_ROUTE = 30     # minimum number of quotes for a route to have a reliable baseline

def T(table_name):
    """Return the fully qualified name of a project table (catalog.schema.table)."""
    return f"{CATALOG}.{SCHEMA}.{table_name}"

print(f"Data Lake base path: {BASE_PATH}")

# COMMAND ----------

# Create the schema (database) and the Volume that hold the Lakehouse
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA} COMMENT 'ABD project - Flight Price Lakehouse'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME} COMMENT 'Data Lake files (landing, bronze, streaming)'")

# Create the folders of the Data Lake
for folder in [LANDING, BRONZE, STREAM_IN, CHECKPOINTS]:
    dbutils.fs.mkdirs(folder)

spark.sql(f"DESCRIBE SCHEMA EXTENDED {CATALOG}.{SCHEMA}").show(truncate=False)

# COMMAND ----------

# Pipeline log: every table written by the notebook is registered here (shown in section 10)
pipeline_log = []

def log_step(layer, name, df_or_count):
    """Register the number of rows produced by a pipeline step."""
    rows = df_or_count if isinstance(df_or_count, int) else df_or_count.count()
    pipeline_log.append((layer, name, rows, datetime.datetime.now().strftime("%H:%M:%S")))
    print(f"✔ [{layer}] {name}: {rows:,} rows")

def save_table(df, table_name, layer, mode="overwrite"):
    """Save a DataFrame as a managed Delta table and log it."""
    (df.write
       .format("delta")
       .mode(mode)
       .option("overwriteSchema", "true")
       .saveAsTable(T(table_name)))
    log_step(layer, T(table_name), spark.table(T(table_name)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upload the source files
# MAGIC Upload the 3 source files to the **landing** folder of the Volume:
# MAGIC *Catalog → workspace → flights → lakehouse → landing → Upload to this volume*
# MAGIC
# MAGIC | File | Where to get it |
# MAGIC |---|---|
# MAGIC | `itineraries_sample.csv` | Sample of `itineraries.csv` from Kaggle *dilwong/flightprices* (created with the helper script `make_sample.py`) |
# MAGIC | `airports.dat` | https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat |
# MAGIC | `airline_reviews.csv` | Kaggle airline reviews dataset (renamed to this file name) |

# COMMAND ----------

# Check that the source files are in the landing folder
display(dbutils.fs.ls(LANDING))

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Bronze layer
# MAGIC In the Bronze layer the data stays organised by **source system**. We only apply **data rules**:
# MAGIC correct data types, remove records without keys, remove duplicates and add ingestion metadata (for audit and reprocessing).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Flight fares (structured data)

# COMMAND ----------

# Read the raw CSV file (all columns as string: the data types are fixed explicitly below)
raw_itin = (spark.read
            .option("header", True)
            .option("escape", '"')
            .csv(FILE_ITINERARIES))

print(f"Raw rows: {raw_itin.count():,} | Columns: {len(raw_itin.columns)}")
raw_itin.printSchema()

# COMMAND ----------

# First look at the content
display(raw_itin.limit(5))

# COMMAND ----------

# Data quality check: nulls / empty strings in the key columns
KEY_COLS = ["legId", "searchDate", "flightDate", "startingAirport", "destinationAirport", "totalFare"]

display(raw_itin.select([count(when(isnull(c) | (trim(col(c)) == ""), True)).alias(c) for c in KEY_COLS]))

# COMMAND ----------

# Data quality check: duplicated quotes (same itinerary, same search day, same fare)
DUP_COLS = ["legId", "searchDate", "fareBasisCode", "totalFare"]

n_total    = raw_itin.count()
n_distinct = raw_itin.dropDuplicates(DUP_COLS).count()
print(f"Duplicated records: {n_total - n_distinct:,} ({(n_total - n_distinct) / n_total:.2%})")

# COMMAND ----------

def try_cast(column_name, data_type):
    """Cast a string column without failing on bad values (invalid values become null).
    Needed because serverless compute runs in ANSI mode, where a plain cast() fails on bad values."""
    return expr(f"try_cast(`{column_name}` AS {data_type})")

TYPE_RULES = {
    "searchDate": "DATE", "flightDate": "DATE",
    "elapsedDays": "INT", "seatsRemaining": "INT", "totalTravelDistance": "INT",
    "baseFare": "DOUBLE", "totalFare": "DOUBLE",
    "isBasicEconomy": "BOOLEAN", "isRefundable": "BOOLEAN", "isNonStop": "BOOLEAN",
}

# Data rules: drop records without keys, drop duplicates, fix data types, keep only valid fares
bz_itin = raw_itin.dropna(subset=KEY_COLS).dropDuplicates(DUP_COLS)

for column_name, data_type in TYPE_RULES.items():
    bz_itin = bz_itin.withColumn(column_name, try_cast(column_name, data_type))

bz_itin = (bz_itin
           .where(col("searchDate").isNotNull() & col("flightDate").isNotNull())
           .where(col("totalFare") > 0)
           .where(col("flightDate") >= col("searchDate"))
           # ingestion metadata (audit / reprocessing)
           .withColumn("source_file", col("_metadata.file_name"))
           .withColumn("ingestion_ts", current_timestamp()))

bz_itin.printSchema()

# COMMAND ----------

# Save the Bronze data as Delta files in the Data Lake, partitioned by search date
# (partitioning allows Spark to read only the days needed: partition pruning)
BZ_ITIN_PATH = f"{BRONZE}/itineraries"

(bz_itin.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("searchDate")
    .save(BZ_ITIN_PATH))

log_step("Bronze", BZ_ITIN_PATH, spark.read.format("delta").load(BZ_ITIN_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Airports (reference data)

# COMMAND ----------

# airports.dat has no header: we define the schema up front (OpenFlights documentation)
airports_schema = StructType([
    StructField("airport_id", IntegerType()), StructField("name", StringType()),
    StructField("city", StringType()),        StructField("country", StringType()),
    StructField("iata", StringType()),        StructField("icao", StringType()),
    StructField("latitude", DoubleType()),    StructField("longitude", DoubleType()),
    StructField("altitude", IntegerType()),   StructField("timezone_offset", StringType()),
    StructField("dst", StringType()),         StructField("tz", StringType()),
    StructField("type", StringType()),        StructField("source", StringType()),
])

bz_airports = (spark.read
               .schema(airports_schema)
               .option("nullValue", "\\N")
               .csv(FILE_AIRPORTS)
               .where(col("iata").isNotNull() & (length("iata") == 3))
               .dropDuplicates(["iata"])
               .select("iata", "name", "city", "country", "latitude", "longitude", "tz")
               .withColumn("ingestion_ts", current_timestamp()))

BZ_AIRPORTS_PATH = f"{BRONZE}/airports"
bz_airports.write.format("delta").mode("overwrite").save(BZ_AIRPORTS_PATH)
log_step("Bronze", BZ_AIRPORTS_PATH, spark.read.format("delta").load(BZ_AIRPORTS_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Passenger reviews (unstructured data)

# COMMAND ----------

# Reviews contain free text with commas, quotes and line breaks: multiLine + escape are needed
raw_reviews = (spark.read
               .option("header", True)
               .option("multiLine", True)
               .option("escape", '"')
               .csv(FILE_REVIEWS))

raw_reviews.printSchema()   # check the column names and adjust REVIEW_*_COL in the config cell if needed

# COMMAND ----------

bz_reviews = (raw_reviews
              .select(col(f"`{REVIEW_AIRLINE_COL}`").alias("airline_name"),
                      col(f"`{REVIEW_TEXT_COL}`").alias("review_text"))
              .dropna()
              .where(length(trim("review_text")) > 20)   # discard empty / meaningless reviews
              .dropDuplicates()
              .withColumn("review_id", monotonically_increasing_id())
              .withColumn("ingestion_ts", current_timestamp()))

BZ_REVIEWS_PATH = f"{BRONZE}/reviews"
bz_reviews.write.format("delta").mode("overwrite").save(BZ_REVIEWS_PATH)
log_step("Bronze", BZ_REVIEWS_PATH, spark.read.format("delta").load(BZ_REVIEWS_PATH))

display(spark.read.format("delta").load(BZ_REVIEWS_PATH).limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Silver layer
# MAGIC In the Silver layer we apply **business rules**: new attributes needed by the business (route, days in advance,
# MAGIC booking window, weekday, main airline, duration in minutes…) and enrichment with reference data (airport cities).
# MAGIC The result is a clean and detailed history of fare quotes, registered as SQL tables.

# COMMAND ----------

# Read the Bronze layer
bz_itin_df     = spark.read.format("delta").load(BZ_ITIN_PATH)
bz_airports_df = spark.read.format("delta").load(BZ_AIRPORTS_PATH)

SEG_SEP = r"\|\|"   # multi-segment fields are separated by "||" in the source

# Reference data for origin and destination (small tables -> broadcast join: copied to every executor,
# so the big fares table is not shuffled across the cluster)
origin_dim = bz_airports_df.select(col("iata").alias("startingAirport"),
                                   col("city").alias("origin_city"),
                                   col("latitude").alias("origin_lat"),
                                   col("longitude").alias("origin_lon"))
dest_dim   = bz_airports_df.select(col("iata").alias("destinationAirport"),
                                   col("city").alias("destination_city"),
                                   col("latitude").alias("destination_lat"),
                                   col("longitude").alias("destination_lon"))

# COMMAND ----------

sv_flights = (bz_itin_df
    # Route and booking window
    .withColumn("route", concat_ws("-", "startingAirport", "destinationAirport"))
    .withColumn("days_in_advance", datediff("flightDate", "searchDate"))
    .withColumn("booking_window",
                when(col("days_in_advance") <= 7,  "1. 0-7 days")
               .when(col("days_in_advance") <= 14, "2. 8-14 days")
               .when(col("days_in_advance") <= 30, "3. 15-30 days")
               .when(col("days_in_advance") <= 60, "4. 31-60 days")
               .otherwise("5. 60+ days"))
    # Calendar attributes of the flight
    .withColumn("flight_weekday", date_format("flightDate", "E"))
    .withColumn("flight_month", date_format("flightDate", "yyyy-MM"))
    .withColumn("is_weekend_flight", dayofweek("flightDate").isin(1, 7))
    # Travel duration: ISO-8601 text (e.g. "PT5H24M" or "P1DT2H") -> minutes
    .withColumn("_d", regexp_extract("travelDuration", r"(\d+)D", 1))
    .withColumn("_h", regexp_extract("travelDuration", r"(\d+)H", 1))
    .withColumn("_m", regexp_extract("travelDuration", r"(\d+)M", 1))
    .withColumn("travel_minutes", expr("coalesce(try_cast(_d AS INT), 0) * 1440 "
                                       "+ coalesce(try_cast(_h AS INT), 0) * 60 "
                                       "+ coalesce(try_cast(_m AS INT), 0)"))
    # Segment information (multi-value fields)
    .withColumn("_airlines", split("segmentsAirlineName", SEG_SEP))
    .withColumn("num_segments", size("_airlines"))
    .withColumn("main_airline", element_at("_airlines", 1))
    .withColumn("is_multi_airline", size(array_distinct("_airlines")) > 1)
    .withColumn("cabin", element_at(split("segmentsCabinCode", SEG_SEP), 1))
    .withColumn("airline_key", lower(element_at(split("main_airline", " "), 1)))   # used to join with reviews
    # Price per mile
    .withColumn("fare_per_mile", when(col("totalTravelDistance") > 0,
                                      round(col("totalFare") / col("totalTravelDistance"), 3)))
    # Enrichment with the airport reference data
    .join(broadcast(origin_dim), "startingAirport", "left")
    .join(broadcast(dest_dim), "destinationAirport", "left")
    .select("legId", "searchDate", "flightDate", "days_in_advance", "booking_window",
            "flight_weekday", "flight_month", "is_weekend_flight",
            "route", "startingAirport", "origin_city", "origin_lat", "origin_lon",
            "destinationAirport", "destination_city", "destination_lat", "destination_lon",
            "main_airline", "airline_key", "is_multi_airline", "num_segments", "isNonStop", "cabin",
            "isBasicEconomy", "isRefundable", "seatsRemaining",
            "travel_minutes", "totalTravelDistance", "baseFare", "totalFare", "fare_per_mile"))

save_table(sv_flights, "sv_flights", "Silver")
display(spark.table(T("sv_flights")).limit(5))

# COMMAND ----------

# Second Silver table, at segment level: one row per flight leg (explode of the "||" lists)
sv_segments = (bz_itin_df
    .select("legId", "searchDate", "totalFare",
            arrays_zip(split("segmentsAirlineName", SEG_SEP).alias("airline"),
                       split("segmentsDepartureAirportCode", SEG_SEP).alias("from_airport"),
                       split("segmentsArrivalAirportCode", SEG_SEP).alias("to_airport"),
                       split("segmentsDistance", SEG_SEP).alias("distance")).alias("segments"))
    .select("legId", "searchDate", "totalFare", posexplode("segments").alias("segment_order", "seg"))
    .select("legId", "searchDate", "totalFare",
            (col("segment_order") + 1).alias("segment_order"),
            col("seg.airline").alias("airline"),
            col("seg.from_airport").alias("from_airport"),
            col("seg.to_airport").alias("to_airport"),
            expr("try_cast(seg.distance AS INT)").alias("distance_miles")))

save_table(sv_segments, "sv_flight_segments", "Silver")
display(spark.table(T("sv_flight_segments")).limit(5))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Quick profile of the Silver table
# MAGIC SELECT COUNT(*)                     AS fare_quotes,
# MAGIC        COUNT(DISTINCT route)        AS routes,
# MAGIC        COUNT(DISTINCT main_airline) AS airlines,
# MAGIC        MIN(searchDate)              AS first_search,
# MAGIC        MAX(searchDate)              AS last_search,
# MAGIC        ROUND(AVG(totalFare), 2)     AS avg_fare_usd
# MAGIC FROM workspace.flights.sv_flights

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Gold layer (structured data)
# MAGIC Each Gold table answers one business question and is ready to be explored with SQL by business users.

# COMMAND ----------

sv = spark.table(T("sv_flights"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.1 Q1: What is the best booking window?
# MAGIC To compare routes with very different prices, we compute a **price index**: average fare of the booking window
# MAGIC divided by the average fare of the route (index 1.00 = route average; 0.85 = 15% cheaper than average).

# COMMAND ----------

w_route = Window.partitionBy("route")

gd_route_advance = (sv
    .groupBy("route", "booking_window")
    .agg(count("*").alias("n_quotes"),
         round(avg("totalFare"), 2).alias("avg_fare"),
         round(percentile_approx("totalFare", 0.5), 2).alias("median_fare"),
         round(min("totalFare"), 2).alias("min_fare"))
    .withColumn("price_index",
                round(col("avg_fare") * sum("n_quotes").over(w_route)
                      / sum(col("avg_fare") * col("n_quotes")).over(w_route), 3)))

save_table(gd_route_advance, "gd_route_advance_price", "Gold")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Q1: average price index per booking window (all routes). Suggested chart: bar chart
# MAGIC SELECT booking_window,
# MAGIC        ROUND(AVG(price_index), 3) AS avg_price_index,
# MAGIC        ROUND(AVG(avg_fare), 2)    AS avg_fare_usd,
# MAGIC        SUM(n_quotes)              AS n_quotes
# MAGIC FROM workspace.flights.gd_route_advance_price
# MAGIC GROUP BY booking_window
# MAGIC ORDER BY booking_window

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2 Q2: Which routes are the cheapest from each airport?

# COMMAND ----------

w_origin = Window.partitionBy("startingAirport").orderBy("avg_fare")

gd_route_ranking = (sv
    .groupBy("route", "startingAirport", "origin_city", "destinationAirport", "destination_city")
    .agg(count("*").alias("n_quotes"),
         round(avg("totalFare"), 2).alias("avg_fare"),
         round(avg("fare_per_mile"), 3).alias("avg_fare_per_mile"),
         round(avg(col("isNonStop").cast("int")) * 100, 1).alias("pct_nonstop"),
         round(avg("travel_minutes"), 0).alias("avg_travel_minutes"))
    .where(col("n_quotes") >= MIN_QUOTES_ROUTE)
    .withColumn("rank_cheapest_from_origin", dense_rank().over(w_origin)))

save_table(gd_route_ranking, "gd_route_ranking", "Gold")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Q2: top 3 cheapest destinations from each airport
# MAGIC SELECT startingAirport, rank_cheapest_from_origin AS rank, destinationAirport, destination_city,
# MAGIC        avg_fare, avg_fare_per_mile, pct_nonstop
# MAGIC FROM workspace.flights.gd_route_ranking
# MAGIC WHERE rank_cheapest_from_origin <= 3
# MAGIC ORDER BY startingAirport, rank

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.3 Q2: Which airlines are the cheapest, and on which weekdays? (pivot)

# COMMAND ----------

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

top_airlines = (sv.groupBy("main_airline").count()
                  .where(col("count") >= MIN_QUOTES_ROUTE))

gd_airline_weekday = (sv
    .join(top_airlines.select("main_airline"), "main_airline")
    .groupBy("main_airline")
    .pivot("flight_weekday", WEEKDAYS)
    .agg(round(avg("totalFare"), 2)))

save_table(gd_airline_weekday, "gd_airline_weekday_fare", "Gold")
display(spark.table(T("gd_airline_weekday_fare")).orderBy("Tue"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.4 Q3: How do prices evolve over time? (7-day moving average)

# COMMAND ----------

# Daily average fare per route, smoothed with a 7-day moving average (window function)
w_moving = Window.partitionBy("route").orderBy("searchDate").rowsBetween(-6, Window.currentRow)

gd_daily_trend = (sv
    .groupBy("route", "searchDate")
    .agg(count("*").alias("n_quotes"), round(avg("totalFare"), 2).alias("avg_fare"))
    .withColumn("moving_avg_7d", round(avg("avg_fare").over(w_moving), 2)))

save_table(gd_daily_trend, "gd_daily_price_trend", "Gold")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Q3: price trend of the 5 routes with most quotes. Suggested chart: line chart (x = searchDate, y = moving_avg_7d, group by route)
# MAGIC WITH top_routes AS (
# MAGIC   SELECT route FROM workspace.flights.gd_daily_price_trend
# MAGIC   GROUP BY route ORDER BY SUM(n_quotes) DESC LIMIT 5
# MAGIC )
# MAGIC SELECT t.searchDate, t.route, t.avg_fare, t.moving_avg_7d
# MAGIC FROM workspace.flights.gd_daily_price_trend t
# MAGIC JOIN top_routes USING (route)
# MAGIC ORDER BY t.searchDate, t.route

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.5 Q4 (preparation): historical price baseline per route
# MAGIC This table is the "memory" used by the streaming process (section 8) to decide if a new quote is a deal.

# COMMAND ----------

gd_baseline = (sv
    .groupBy("route", "startingAirport", "destinationAirport")
    .agg(count("*").alias("n_quotes"),
         round(avg("totalFare"), 2).alias("avg_fare"),
         round(stddev("totalFare"), 2).alias("std_fare"),
         round(percentile_approx("totalFare", 0.25), 2).alias("p25_fare"),
         round(min("totalFare"), 2).alias("min_fare"))
    .where(col("n_quotes") >= MIN_QUOTES_ROUTE))

save_table(gd_baseline, "gd_route_price_baseline", "Gold")
display(spark.table(T("gd_route_price_baseline")).orderBy(desc("n_quotes")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Unstructured data: passenger reviews (Q5)
# MAGIC We apply the **Word Count** algorithm (ABD06) to the review text: clean the text, split it into words (`explode`),
# MAGIC remove stop words and count. We then measure the share of reviews that mention typical complaint topics and
# MAGIC cross it with the average fare of each airline (structured data).

# COMMAND ----------

STOP_WORDS = ["the", "and", "was", "were", "for", "that", "with", "this", "they", "but", "not", "had", "have",
              "from", "our", "you", "are", "flight", "flights", "airline", "airlines", "very", "which", "there",
              "would", "their", "when", "all", "after", "been", "one", "out", "get", "got", "did", "has", "even",
              "only", "time", "again", "will", "could", "also", "just", "about", "them", "then", "than", "she",
              "her", "him", "his", "who", "what", "any", "into", "back", "because", "over", "being", "verified",
              "trip", "review", "can", "seat", "seats", "plane", "told", "said", "way"]

COMPLAINT_TERMS = ["delay", "delayed", "cancel", "cancelled", "canceled", "lost", "refund", "rude",
                   "late", "dirty", "broken", "missed", "stranded", "worst", "terrible"]

bz_reviews_df = spark.read.format("delta").load(BZ_REVIEWS_PATH)

# Silver: one row per (review, word) after cleaning
sv_review_words = (bz_reviews_df
    .withColumn("airline_key", lower(element_at(split("airline_name", " "), 1)))
    .withColumn("clean_text", regexp_replace(lower("review_text"), "[^a-z\\s]", " "))
    .select("review_id", "airline_key", "airline_name",
            explode(split("clean_text", "\\s+")).alias("word"))
    .where((length("word") > 2) & (~col("word").isin(STOP_WORDS))))

save_table(sv_review_words, "sv_review_words", "Silver")

# COMMAND ----------

# Gold: top 10 words per airline (row_number window)
w_airline_words = Window.partitionBy("airline_key").orderBy(desc("count"), "word")

gd_top_words = (spark.table(T("sv_review_words"))
    .groupBy("airline_key", "word").count()
    .withColumn("rank", row_number().over(w_airline_words))
    .where(col("rank") <= 10))

save_table(gd_top_words, "gd_airline_top_words", "Gold")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Most frequent words per airline (only airlines that also exist in the fares data)
# MAGIC SELECT w.airline_key, w.rank, w.word, w.count
# MAGIC FROM workspace.flights.gd_airline_top_words w
# MAGIC WHERE w.airline_key IN (SELECT DISTINCT airline_key FROM workspace.flights.sv_flights)
# MAGIC ORDER BY w.airline_key, w.rank

# COMMAND ----------

# Gold: complaints vs price per airline (unstructured + structured data)
complaint_regex = "\\b(" + "|".join(COMPLAINT_TERMS) + ")\\b"

complaints = (bz_reviews_df
    .withColumn("airline_key", lower(element_at(split("airline_name", " "), 1)))
    .withColumn("has_complaint", lower("review_text").rlike(complaint_regex))
    .groupBy("airline_key")
    .agg(count("*").alias("n_reviews"),
         round(avg(col("has_complaint").cast("int")) * 100, 1).alias("pct_reviews_with_complaint")))

airline_prices = (spark.table(T("sv_flights"))
    .groupBy("airline_key")
    .agg(first("main_airline").alias("airline"),
         count("*").alias("n_quotes"),
         round(avg("totalFare"), 2).alias("avg_fare"),
         round(avg("fare_per_mile"), 3).alias("avg_fare_per_mile")))

gd_price_vs_complaints = airline_prices.join(complaints, "airline_key", "inner")

save_table(gd_price_vs_complaints, "gd_airline_price_vs_complaints", "Gold")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Q5: are cheaper airlines the ones with more complaints? Suggested chart: scatter (x = avg_fare_per_mile, y = pct_reviews_with_complaint)
# MAGIC SELECT airline, avg_fare, avg_fare_per_mile, n_reviews, pct_reviews_with_complaint
# MAGIC FROM workspace.flights.gd_airline_price_vs_complaints
# MAGIC ORDER BY avg_fare_per_mile

# COMMAND ----------

# MAGIC %md
# MAGIC # 8. Streaming data: real-time price alerts (Q4)
# MAGIC Partners (airlines and travel agencies) continuously send new fare quotes as JSON files to a folder.
# MAGIC A **Spark Structured Streaming** query:
# MAGIC 1. **Source:** reads the new files from `stream_input` (each file is read only once, thanks to the checkpoint);
# MAGIC 2. **Transformation:** joins each quote with the historical baseline of its route (stream-static join) and flags quotes
# MAGIC    at least 30% cheaper than average as **deals**;
# MAGIC 3. **Sink:** appends the result to a Delta table (`st_price_quotes`).
# MAGIC
# MAGIC > **Databricks Free Edition (serverless):** continuous triggers and the `memory` sink are not available, so we use
# MAGIC > `trigger(availableNow=True)`: each run processes all the files that arrived since the last run and then stops (incremental micro-batches).

# COMMAND ----------

# Clean the streaming folders (fresh start for the demo)
dbutils.fs.rm(STREAM_IN, True)
dbutils.fs.rm(CHECKPOINTS, True)
dbutils.fs.mkdirs(STREAM_IN)
spark.sql(f"DROP TABLE IF EXISTS {T('st_price_quotes')}")
spark.sql(f"DROP TABLE IF EXISTS {T('st_deals_by_route')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.1 Quote generator (simulates the partners)

# COMMAND ----------

# Real routes and airlines from the Gold/Silver layers, so the simulated quotes are realistic
routes = [r.asDict() for r in (spark.table(T("gd_route_price_baseline"))
                                    .orderBy(desc("n_quotes")).limit(40).collect())]
airlines = [r["main_airline"] for r in (spark.table(T("sv_flights"))
                                            .groupBy("main_airline").count()
                                            .orderBy(desc("count")).limit(8).collect())]
PARTNERS = ["Expedia", "Kayak", "eDreams", "Airline direct", "Trip.com"]

def generate_quote_file(n_quotes=25, deal_probability=0.15):
    """Write one JSON file with n_quotes new fare quotes to the streaming input folder."""
    now = datetime.datetime.now()
    quotes = []
    for _ in range(n_quotes):
        r = random.choice(routes)
        std = r["std_fare"] or r["avg_fare"] * 0.2
        if random.random() < deal_probability:
            fare = r["avg_fare"] * random.uniform(0.40, 0.70)       # a real deal
        else:
            fare = py.max(29.0, random.gauss(r["avg_fare"], std))      # normal market price
        quotes.append({
            "quote_id": str(uuid.uuid4()),
            "quote_ts": (now + datetime.timedelta(seconds=random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S"),
            "route": r["route"],
            "startingAirport": r["startingAirport"],
            "destinationAirport": r["destinationAirport"],
            "airline": random.choice(airlines),
            "partner": random.choice(PARTNERS),
            "flightDate": (now + datetime.timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d"),
            "totalFare": py.round(fare, 2),
        })
    file_name = f"{STREAM_IN}/quotes_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    dbutils.fs.put(file_name, "\n".join(json.dumps(q) for q in quotes), True)
    return file_name

# First wave: 3 files arrive
for _ in range(3):
    print("New file:", generate_quote_file())

# COMMAND ----------

display(dbutils.fs.ls(STREAM_IN))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.2 Streaming query: source → transformation → sink

# COMMAND ----------

# Source: file streams need the schema up front
quote_schema = StructType([
    StructField("quote_id", StringType()),       StructField("quote_ts", StringType()),
    StructField("route", StringType()),          StructField("startingAirport", StringType()),
    StructField("destinationAirport", StringType()), StructField("airline", StringType()),
    StructField("partner", StringType()),        StructField("flightDate", StringType()),
    StructField("totalFare", DoubleType()),
])

quotes_stream_raw = (spark.readStream
                     .format("json")
                     .schema(quote_schema)
                     .option("maxFilesPerTrigger", 1)    # one file per micro-batch
                     .load(STREAM_IN))

# Transformation: stream-static join with the historical baseline (Gold) + deal rule
baseline = spark.table(T("gd_route_price_baseline")).select("route", "avg_fare", "p25_fare")

quotes_stream = (quotes_stream_raw
    .withColumn("quote_ts", to_timestamp("quote_ts"))
    .withColumn("flightDate", to_date("flightDate"))
    .join(baseline, "route", "left")
    .withColumn("pct_vs_avg", round((col("totalFare") / col("avg_fare") - 1) * 100, 1))
    .withColumn("is_deal", col("totalFare") <= col("avg_fare") * (1 - DEAL_THRESHOLD))
    .withColumn("processed_ts", current_timestamp()))

quotes_stream.printSchema()
print("Is streaming:", quotes_stream.isStreaming)

# COMMAND ----------

def run_quotes_stream():
    """Sink: process all new files and append them to the Delta table st_price_quotes."""
    query = (quotes_stream.writeStream
             .queryName("q_price_quotes")
             .format("delta")
             .outputMode("append")
             .option("checkpointLocation", f"{CHECKPOINTS}/price_quotes")
             .trigger(availableNow=True)
             .toTable(T("st_price_quotes")))
    query.awaitTermination()
    batches = query.recentProgress
    print(f"Micro-batches: {len(batches)} | rows processed in this run: {py.sum(b['numInputRows'] for b in batches)}")
    return query

q_quotes = run_quotes_stream()
log_step("Streaming", T("st_price_quotes") + " (run 1)", spark.table(T("st_price_quotes")))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Price alerts: quotes at least 30% below the route average
# MAGIC SELECT quote_ts, route, airline, partner, flightDate, totalFare, avg_fare, pct_vs_avg
# MAGIC FROM workspace.flights.st_price_quotes
# MAGIC WHERE is_deal
# MAGIC ORDER BY pct_vs_avg

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.3 Streaming aggregation (complete mode)
# MAGIC A second query keeps an always up-to-date summary of quotes and deals per route.

# COMMAND ----------

def run_deals_by_route_stream():
    deals_by_route = (quotes_stream
        .groupBy("route")
        .agg(count("*").alias("n_quotes"),
             sum(col("is_deal").cast("int")).alias("n_deals"),
             min("totalFare").alias("best_fare")))
    query = (deals_by_route.writeStream
             .queryName("q_deals_by_route")
             .format("delta")
             .outputMode("complete")          # the whole result table is rewritten at each micro-batch
             .option("checkpointLocation", f"{CHECKPOINTS}/deals_by_route")
             .trigger(availableNow=True)
             .toTable(T("st_deals_by_route")))
    query.awaitTermination()
    return query

q_deals = run_deals_by_route_stream()
display(spark.table(T("st_deals_by_route")).orderBy(desc("n_deals")))

# COMMAND ----------

# MAGIC %md
# MAGIC # 9. Delta Lake: "best price board" with MERGE and time travel
# MAGIC The Gold table `gd_best_price_board` keeps the **best price found so far for each route** (what Skyscanner shows as
# MAGIC "cheapest price"). Every time new quotes are processed it is updated with a `MERGE` (upsert):
# MAGIC routes already on the board are updated only if the new price is lower; new routes are inserted.
# MAGIC Each operation creates a new version of the Delta table, which allows auditing with `DESCRIBE HISTORY` and **time travel**.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {T('gd_best_price_board')} (
  route       STRING,
  airline     STRING,
  partner     STRING,
  flightDate  DATE,
  best_fare   DOUBLE,
  quote_ts    TIMESTAMP,
  updated_ts  TIMESTAMP
) USING DELTA
""")

def update_best_price_board():
    """MERGE the cheapest streamed quote of each route into the best price board."""
    w_best = Window.partitionBy("route").orderBy("totalFare", desc("quote_ts"))
    (spark.table(T("st_price_quotes"))
         .withColumn("rn", row_number().over(w_best))
         .where("rn = 1")
         .select("route", "airline", "partner", "flightDate", col("totalFare").alias("best_fare"), "quote_ts")
         .createOrReplaceTempView("v_best_quotes"))
    spark.sql(f"""
        MERGE INTO {T('gd_best_price_board')} AS t
        USING v_best_quotes AS s
        ON t.route = s.route
        WHEN MATCHED AND s.best_fare < t.best_fare THEN UPDATE SET
            t.airline = s.airline, t.partner = s.partner, t.flightDate = s.flightDate,
            t.best_fare = s.best_fare, t.quote_ts = s.quote_ts, t.updated_ts = current_timestamp()
        WHEN NOT MATCHED THEN INSERT (route, airline, partner, flightDate, best_fare, quote_ts, updated_ts)
            VALUES (s.route, s.airline, s.partner, s.flightDate, s.best_fare, s.quote_ts, current_timestamp())
    """).display()   # shows rows inserted / updated

update_best_price_board()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.1 Second wave: new quotes arrive
# MAGIC We simulate 3 more files. Running the same streaming query again shows that **only the new files are processed**
# MAGIC (the checkpoint guarantees each file is processed exactly once), and the board is updated with a new MERGE.

# COMMAND ----------

rows_before = spark.table(T("st_price_quotes")).count()

for _ in range(3):
    print("New file:", generate_quote_file(deal_probability=0.25))

q_quotes = run_quotes_stream()
q_deals  = run_deals_by_route_stream()

rows_after = spark.table(T("st_price_quotes")).count()
print(f"Rows before: {rows_before} | after: {rows_after} | new rows: {rows_after - rows_before}")
log_step("Streaming", T("st_price_quotes") + " (run 2)", rows_after)

update_best_price_board()
log_step("Gold", T("gd_best_price_board"), spark.table(T("gd_best_price_board")))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Audit: every write to the board is a new version of the Delta table
# MAGIC DESCRIBE HISTORY workspace.flights.gd_best_price_board

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Time travel: compare the board after the 1st wave (version 1) with the current version
# MAGIC SELECT cur.route,
# MAGIC        v1.best_fare  AS best_fare_after_wave_1,
# MAGIC        cur.best_fare AS best_fare_now,
# MAGIC        ROUND(cur.best_fare - v1.best_fare, 2) AS change_usd,
# MAGIC        cur.airline, cur.partner
# MAGIC FROM workspace.flights.gd_best_price_board AS cur
# MAGIC LEFT JOIN workspace.flights.gd_best_price_board VERSION AS OF 1 AS v1
# MAGIC        ON cur.route = v1.route
# MAGIC ORDER BY change_usd NULLS FIRST

# COMMAND ----------

# MAGIC %md
# MAGIC # 10. Execution summary

# COMMAND ----------

# Rows produced by each step of the pipeline (Bronze -> Silver -> Gold -> Streaming)
display(spark.createDataFrame(pipeline_log, ["layer", "object", "rows", "time"]))

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN workspace.flights

# COMMAND ----------

# MAGIC %md
# MAGIC # 11. Conclusions
# MAGIC > ✏️ *Fill in with the actual numbers after running the notebook.*
# MAGIC
# MAGIC ### Answers to the business questions
# MAGIC | # | Question | Main finding |
# MAGIC |---|---|---|
# MAGIC | Q1 | Best booking window | _e.g. "Fares bought 31–60 days in advance are on average X% below the route average; last-minute (0–7 days) are Y% above."_ |
# MAGIC | Q2 | Cheapest routes / airlines / weekdays | _e.g. "From each airport, the cheapest destinations are …; airline … is the cheapest per mile; Tuesdays/Wednesdays are …"_ |
# MAGIC | Q3 | Price evolution | _e.g. "Prices on the 5 busiest routes peak in … and drop in …"_ |
# MAGIC | Q4 | Real-time deals | _e.g. "The streaming pipeline evaluated N quotes in 2 waves, detected K deals (≥30% below average) and kept the best price per route updated with MERGE."_ |
# MAGIC | Q5 | Complaints vs price | _e.g. "Low-cost airlines (…) show X% of reviews with complaints vs Y% for …; most frequent complaint words: delay, refund…"_ |
# MAGIC
# MAGIC ### Big Data technologies used
# MAGIC | Concept (class) | Where in the notebook |
# MAGIC |---|---|
# MAGIC | Spark DataFrames, transformations and actions (ABD04) | Whole pipeline |
# MAGIC | Spark SQL, views/tables, aggregations, `pivot`, `explode`, window functions (ABD05/06) | Sections 5, 6, 7 |
# MAGIC | Word Count on unstructured text (ABD06) | Section 7 |
# MAGIC | Data Lakehouse, Medallion architecture, Delta Lake (`MERGE`, history, time travel) (ABD07) | Sections 4–6, 9 |
# MAGIC | Structured Streaming: file source, stream-static join, append and complete modes, checkpoints (ABD08) | Section 8 |
# MAGIC | Distributed processing: partitioning by date, broadcast join (ABD03) | Sections 4.1, 5 |
# MAGIC
# MAGIC ### Limitations and next steps
# MAGIC * We worked on a sample of the full dataset (Free Edition limits); the code scales to the full ~30 GB on a larger cluster.
# MAGIC * The streaming quotes are simulated; in production they would come from partner APIs or Kafka.
# MAGIC * The data covers US domestic flights in 2022; the same pipeline could be applied to European routes (e.g. from Lisbon).
# MAGIC * Next steps: a price prediction model (Spark MLlib) and sentiment analysis of the reviews.

# COMMAND ----------

# MAGIC %md
# MAGIC ### (Optional) Clean the environment
# MAGIC Uncomment and run only when the project has been delivered.

# COMMAND ----------

# spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")