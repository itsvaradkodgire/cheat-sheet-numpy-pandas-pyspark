# PySpark, Pandas, SQL & ANSI SQL Cheat Sheet for Data Analysis

## Table of Contents
1. [SQL & ANSI SQL Fundamentals](#sql--ansi-sql-fundamentals)
2. [PySpark DataFrame Operations](#pyspark-dataframe-operations)
3. [Pandas Operations](#pandas-operations)
4. [PySpark ETL Patterns](#pyspark-etl-patterns)
5. [Common Query Patterns](#common-query-patterns)

---

## SQL & ANSI SQL Fundamentals

### Database & Table Operations
```sql
-- Database Operations
CREATE DATABASE dac_dbt;
USE dac_dbt;
DROP DATABASE IF EXISTS dac_dbt;

-- Table Creation with Constraints
CREATE TABLE emp (
    empcode VARCHAR(15) PRIMARY KEY,
    empname VARCHAR(60),
    deptcode VARCHAR(15),
    birthdate DATE NOT NULL,
    joindate DATE NOT NULL,
    sex CHAR(1) CHECK (sex IN ('M', 'F', 'T')),
    basicpay INTEGER,
    FOREIGN KEY (deptcode) REFERENCES dept(deptcode)
);

-- Add Constraints Later
ALTER TABLE emp ADD PRIMARY KEY (empcode);
ALTER TABLE emp ADD FOREIGN KEY (deptcode) REFERENCES dept(deptcode);
```

### Basic Queries
```sql
-- Select with Conditions
SELECT empname, empcode, desigcode FROM emp;
SELECT * FROM dept WHERE budget > 20;
SELECT empname FROM emp WHERE basicpay BETWEEN 11000 AND 12000;
SELECT empname FROM emp WHERE empname LIKE '%i' OR empname LIKE '%y';

-- Sorting
SELECT * FROM emp ORDER BY birthdate ASC;
SELECT * FROM emp ORDER BY basicpay DESC, empname ASC;

-- Distinct Values
SELECT DISTINCT deptcode FROM emp;
```

### Joins
```sql
-- Inner Join
SELECT e.empname, d.deptname 
FROM emp e 
INNER JOIN dept d ON e.deptcode = d.deptcode;

-- Self Join (Employee-Supervisor)
SELECT e.empname AS employee, s.empname AS supervisor
FROM emp e 
LEFT JOIN emp s ON e.supcode = s.empcode;

-- Multiple Joins
SELECT e.empname, d.deptname, des.designame
FROM emp e
JOIN dept d ON e.deptcode = d.deptcode
JOIN desig des ON e.desigcode = des.desigcode;
```

### Aggregation Functions
```sql
-- Basic Aggregations
SELECT COUNT(*) FROM emp;
SELECT AVG(basicpay) FROM emp;
SELECT MAX(basicpay), MIN(basicpay) FROM emp;
SELECT SUM(budget) FROM dept;

-- Group By
SELECT deptcode, COUNT(*) as emp_count 
FROM emp 
GROUP BY deptcode;

SELECT deptcode, AVG(basicpay) as avg_salary
FROM emp 
GROUP BY deptcode
HAVING AVG(basicpay) > 12000;
```

### Subqueries
```sql
-- Scalar Subquery
SELECT empname FROM emp 
WHERE basicpay > (SELECT AVG(basicpay) FROM emp);

-- IN Subquery
SELECT empname FROM emp 
WHERE deptcode IN (SELECT deptcode FROM dept WHERE budget > 30);

-- EXISTS Subquery
SELECT empname FROM emp e
WHERE EXISTS (SELECT 1 FROM salary s WHERE s.empcode = e.empcode);

-- Correlated Subquery
SELECT empname, basicpay FROM emp e1
WHERE basicpay = (SELECT MAX(basicpay) FROM emp e2 WHERE e2.deptcode = e1.deptcode);
```

### Window Functions (ANSI SQL)
```sql
-- Ranking Functions
SELECT empname, basicpay,
       ROW_NUMBER() OVER (ORDER BY basicpay DESC) as rank_num,
       RANK() OVER (ORDER BY basicpay DESC) as rank_dense,
       DENSE_RANK() OVER (ORDER BY basicpay DESC) as dense_rank
FROM emp;

-- Partition By
SELECT empname, deptcode, basicpay,
       ROW_NUMBER() OVER (PARTITION BY deptcode ORDER BY basicpay DESC) as dept_rank
FROM emp;

-- Aggregate Window Functions
SELECT empname, basicpay,
       AVG(basicpay) OVER () as overall_avg,
       AVG(basicpay) OVER (PARTITION BY deptcode) as dept_avg
FROM emp;

-- Lead/Lag
SELECT empname, joindate,
       LAG(joindate) OVER (ORDER BY joindate) as prev_join_date,
       LEAD(joindate) OVER (ORDER BY joindate) as next_join_date
FROM emp;
```

### Date Functions
```sql
-- Date Calculations
SELECT empname, 
       YEAR(CURDATE()) - YEAR(birthdate) as age,
       DATEDIFF(CURDATE(), joindate) / 365 as experience_years
FROM emp;

-- Date Formatting
SELECT empname, DATE_FORMAT(birthdate, '%Y-%m-%d') as formatted_date
FROM emp;
```

---

## PySpark DataFrame Operations

### Setup and Data Loading
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Create Spark Session
spark = SparkSession.builder.appName("DataAnalysis").getOrCreate()

# Load Data
df = spark.read.csv("file.csv", header=True, inferSchema=True)
df = spark.read.parquet("file.parquet")
df = spark.read.jdbc(url="jdbc:mysql://localhost:3306/db", 
                     table="emp", properties={"user":"root","password":"pwd"})
```

### Basic Operations
```python
# Show data
df.show()
df.show(20, truncate=False)
df.printSchema()
df.describe().show()

# Select columns
df.select("empname", "basicpay").show()
df.select(col("empname").alias("employee_name")).show()

# Filter rows
df.filter(col("basicpay") > 12000).show()
df.filter((col("sex") == "F") & (col("basicpay") > 10000)).show()
df.where("deptcode = 'ACCT'").show()

# Sort
df.orderBy("basicpay", ascending=False).show()
df.orderBy(col("basicpay").desc(), col("empname").asc()).show()
```

### Joins
```python
# Join DataFrames
emp_df = spark.table("emp")
dept_df = spark.table("dept")

# Inner Join
result = emp_df.join(dept_df, "deptcode", "inner")

# Left Join with different column names
result = emp_df.alias("e").join(
    dept_df.alias("d"), 
    col("e.deptcode") == col("d.deptcode"), 
    "left"
)

# Self Join
supervisor_df = emp_df.alias("s")
result = emp_df.alias("e").join(
    supervisor_df, 
    col("e.supcode") == col("s.empcode"), 
    "left"
).select(
    col("e.empname").alias("employee"),
    col("s.empname").alias("supervisor")
)
```

### Aggregations
```python
# Basic aggregations
df.agg(
    count("*").alias("total_count"),
    avg("basicpay").alias("avg_salary"),
    max("basicpay").alias("max_salary"),
    min("basicpay").alias("min_salary")
).show()

# Group By
df.groupBy("deptcode").agg(
    count("*").alias("emp_count"),
    avg("basicpay").alias("avg_salary"),
    max("basicpay").alias("max_salary")
).show()

# Multiple grouping
df.groupBy("deptcode", "sex").agg(
    count("*").alias("count"),
    avg("basicpay").alias("avg_salary")
).show()
```

### Window Functions
```python
from pyspark.sql.window import Window

# Define window spec
windowSpec = Window.orderBy(col("basicpay").desc())
windowSpecPartitioned = Window.partitionBy("deptcode").orderBy(col("basicpay").desc())

# Ranking functions
df.select(
    "*",
    row_number().over(windowSpec).alias("rank"),
    dense_rank().over(windowSpec).alias("dense_rank"),
    rank().over(windowSpec).alias("rank_with_gaps")
).show()

# Department-wise ranking
df.select(
    "*",
    row_number().over(windowSpecPartitioned).alias("dept_rank")
).show()

# Moving averages
df.select(
    "*",
    avg("basicpay").over(windowSpec.rowsBetween(-2, 0)).alias("moving_avg_3")
).show()
```

### Advanced Functions
```python
# String functions
df.select(
    upper("empname").alias("name_upper"),
    length("empname").alias("name_length"),
    substring("empname", 1, 3).alias("name_first_3")
).show()

# Date functions
df.select(
    "*",
    year(current_date()) - year("birthdate").alias("age"),
    datediff(current_date(), "joindate").alias("days_since_joining"),
    months_between(current_date(), "joindate").alias("months_experience")
).show()

# Conditional logic
df.select(
    "*",
    when(col("basicpay") > 15000, "High")
    .when(col("basicpay") > 10000, "Medium")
    .otherwise("Low").alias("salary_category")
).show()

# Null handling
df.select("*").na.drop().show()  # Drop rows with any null
df.na.fill({"basicpay": 0, "empname": "Unknown"}).show()  # Fill nulls
```

---

## Pandas Operations

### Data Loading and Basic Operations
```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("file.csv")
df = pd.read_sql("SELECT * FROM emp", connection)
df = pd.read_parquet("file.parquet")

# Basic info
df.head()
df.info()
df.describe()
df.shape
df.columns
```

### Data Selection and Filtering
```python
# Select columns
df['empname']
df[['empname', 'basicpay']]

# Filter rows
df[df['basicpay'] > 12000]
df[(df['sex'] == 'F') & (df['basicpay'] > 10000)]
df.query("deptcode == 'ACCT'")
df.query("basicpay > 12000 and sex == 'F'")

# Sorting
df.sort_values('basicpay', ascending=False)
df.sort_values(['deptcode', 'basicpay'], ascending=[True, False])
```

### Joins and Merging
```python
# Merge DataFrames
emp_df = pd.read_csv("emp.csv")
dept_df = pd.read_csv("dept.csv")

# Inner join
result = emp_df.merge(dept_df, on='deptcode', how='inner')

# Left join with different column names
result = emp_df.merge(dept_df, left_on='deptcode', right_on='dept_id', how='left')

# Self join (employee-supervisor)
supervisor_df = emp_df[['empcode', 'empname']].rename(
    columns={'empcode': 'supcode', 'empname': 'supervisor_name'}
)
result = emp_df.merge(supervisor_df, on='supcode', how='left')
```

### Aggregations and Grouping
```python
# Basic aggregations
df['basicpay'].sum()
df['basicpay'].mean()
df['basicpay'].max()
df['basicpay'].min()
df['basicpay'].count()

# Group by operations
df.groupby('deptcode')['basicpay'].mean()
df.groupby('deptcode').agg({
    'basicpay': ['mean', 'max', 'min', 'count'],
    'empcode': 'count'
})

# Multiple grouping
df.groupby(['deptcode', 'sex'])['basicpay'].mean()

# Custom aggregations
df.groupby('deptcode').agg({
    'basicpay': lambda x: x.max() - x.min(),  # Range
    'empname': lambda x: ', '.join(x)  # Concatenate names
})
```

### Window Functions and Advanced Operations
```python
# Ranking
df['rank'] = df['basicpay'].rank(method='dense', ascending=False)
df['dept_rank'] = df.groupby('deptcode')['basicpay'].rank(method='dense', ascending=False)

# Rolling calculations
df['moving_avg_3'] = df['basicpay'].rolling(window=3).mean()

# Percentage calculations
df['salary_pct'] = df['basicpay'] / df['basicpay'].sum() * 100

# Cumulative calculations
df['cumulative_salary'] = df['basicpay'].cumsum()

# Apply custom functions
df['age'] = df['birthdate'].apply(lambda x: 2024 - pd.to_datetime(x).year)

# String operations
df['empname_upper'] = df['empname'].str.upper()
df['name_length'] = df['empname'].str.len()
df['ends_with_i'] = df['empname'].str.endswith('i')
```

---

## PySpark ETL Patterns

### Data Ingestion
```python
# Read from various sources
def read_data_source(spark, source_type, path, **kwargs):
    if source_type == "csv":
        return spark.read.csv(path, header=True, inferSchema=True, **kwargs)
    elif source_type == "parquet":
        return spark.read.parquet(path)
    elif source_type == "json":
        return spark.read.json(path)
    elif source_type == "delta":
        return spark.read.format("delta").load(path)
    elif source_type == "jdbc":
        return spark.read.jdbc(**kwargs)

# Read multiple files
df = spark.read.csv("data/*/emp_*.csv", header=True, inferSchema=True)
```

### Data Transformation
```python
# Schema validation and casting
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType

expected_schema = StructType([
    StructField("empcode", StringType(), False),
    StructField("empname", StringType(), False),
    StructField("basicpay", IntegerType(), True),
    StructField("birthdate", DateType(), True)
])

df = df.select([col(c).cast(expected_schema[c].dataType) for c in expected_schema.names])

# Data cleaning
def clean_employee_data(df):
    return df \
        .filter(col("empcode").isNotNull()) \
        .filter(col("basicpay") > 0) \
        .withColumn("empname", trim(upper(col("empname")))) \
        .withColumn("age", year(current_date()) - year("birthdate")) \
        .dropDuplicates(["empcode"])

# Data enrichment
def enrich_employee_data(df, dept_df, grade_df):
    return df \
        .join(dept_df, "deptcode", "left") \
        .join(grade_df, ["gradecode", "gradelevel"], "left") \
        .withColumn("experience_years", 
                   months_between(current_date(), "joindate") / 12) \
        .withColumn("salary_grade", 
                   when(col("basicpay") > 15000, "Senior")
                   .when(col("basicpay") > 10000, "Mid")
                   .otherwise("Junior"))
```

### Data Quality Checks
```python
def data_quality_checks(df, table_name):
    checks = {}
    
    # Row count
    checks[f"{table_name}_row_count"] = df.count()
    
    # Null checks
    null_counts = df.select([
        sum(col(c).isNull().cast("int")).alias(c) 
        for c in df.columns
    ]).collect()[0].asDict()
    
    # Duplicate checks
    total_rows = df.count()
    unique_rows = df.dropDuplicates().count()
    checks[f"{table_name}_duplicates"] = total_rows - unique_rows
    
    # Business rule checks
    checks[f"{table_name}_invalid_salary"] = df.filter(col("basicpay") <= 0).count()
    checks[f"{table_name}_future_birthdates"] = df.filter(
        col("birthdate") > current_date()
    ).count()
    
    return checks

# Data profiling
def profile_dataframe(df):
    profile = {}
    for column in df.columns:
        col_stats = df.select(
            count(col(column)).alias("count"),
            countDistinct(col(column)).alias("distinct"),
            sum(col(column).isNull().cast("int")).alias("nulls")
        ).collect()[0]
        
        profile[column] = {
            "count": col_stats["count"],
            "distinct": col_stats["distinct"],
            "nulls": col_stats["nulls"],
            "null_percentage": col_stats["nulls"] / col_stats["count"] * 100
        }
    
    return profile
```

### Performance Optimization
```python
# Caching
df.cache()
df.persist(StorageLevel.MEMORY_AND_DISK)

# Partitioning
df.repartition(10)  # Increase partitions
df.coalesce(2)      # Decrease partitions
df.repartition("deptcode")  # Partition by column

# Bucketing for joins
df.write \
  .bucketBy(10, "deptcode") \
  .sortBy("empcode") \
  .saveAsTable("emp_bucketed")

# Broadcast joins for small tables
from pyspark.sql.functions import broadcast
large_df.join(broadcast(small_df), "key")

# Column pruning
df.select("empcode", "empname", "basicpay")  # Only select needed columns

# Predicate pushdown
df.filter("deptcode = 'ACCT'").select("empname", "basicpay")
```

### Data Output
```python
# Write to different formats
def write_data(df, output_path, format_type, mode="overwrite", **kwargs):
    writer = df.write.mode(mode)
    
    if format_type == "parquet":
        writer.parquet(output_path, **kwargs)
    elif format_type == "delta":
        writer.format("delta").save(output_path, **kwargs)
    elif format_type == "csv":
        writer.option("header", "true").csv(output_path, **kwargs)
    elif format_type == "json":
        writer.json(output_path, **kwargs)

# Partitioned writes
df.write \
  .partitionBy("deptcode", "year") \
  .parquet("output/emp_partitioned")

# Database writes
df.write \
  .format("jdbc") \
  .option("url", "jdbc:postgresql://localhost:5432/db") \
  .option("dbtable", "emp_processed") \
  .option("user", "username") \
  .option("password", "password") \
  .save()
```

---

## Common Query Patterns

### Ranking and Top N
```sql
-- SQL: Top N employees by salary
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (ORDER BY basicpay DESC) as rn
    FROM emp
) WHERE rn <= 5;

-- SQL: Top employee in each department
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY deptcode ORDER BY basicpay DESC) as rn
    FROM emp
) WHERE rn = 1;
```

```python
# PySpark: Top N employees
from pyspark.sql.window import Window

windowSpec = Window.orderBy(col("basicpay").desc())
top_employees = df.select(
    "*",
    row_number().over(windowSpec).alias("rank")
).filter(col("rank") <= 5)

# PySpark: Top employee in each department
windowSpec = Window.partitionBy("deptcode").orderBy(col("basicpay").desc())
top_by_dept = df.select(
    "*",
    row_number().over(windowSpec).alias("dept_rank")
).filter(col("dept_rank") == 1)
```

```python
# Pandas: Top N employees
top_employees = df.nlargest(5, 'basicpay')

# Pandas: Top employee in each department
top_by_dept = df.groupby('deptcode').apply(lambda x: x.nlargest(1, 'basicpay'))
```

### Time Series and Date Analysis
```sql
-- SQL: Experience calculation
SELECT empname, 
       DATEDIFF(CURDATE(), joindate) / 365 as experience_years
FROM emp
WHERE DATEDIFF(CURDATE(), joindate) / 365 >= 25;
```

```python
# PySpark: Experience calculation
df.select(
    "*",
    (datediff(current_date(), "joindate") / 365).alias("experience_years")
).filter(col("experience_years") >= 25)

# Pandas: Experience calculation
df['experience_years'] = (pd.Timestamp.now() - pd.to_datetime(df['joindate'])).dt.days / 365
experienced = df[df['experience_years'] >= 25]
```

### Complex Aggregations
```sql
-- SQL: Department statistics
SELECT d.deptname,
       COUNT(e.empcode) as emp_count,
       AVG(e.basicpay) as avg_salary,
       MAX(e.basicpay) as max_salary,
       SUM(e.basicpay) as total_salary,
       d.budget,
       (SUM(e.basicpay) / d.budget * 100) as budget_utilization
FROM dept d
LEFT JOIN emp e ON d.deptcode = e.deptcode
GROUP BY d.deptcode, d.deptname, d.budget;
```

```python
# PySpark: Department statistics
dept_stats = emp_df.join(dept_df, "deptcode", "left") \
    .groupBy("deptcode", "deptname", "budget") \
    .agg(
        count("empcode").alias("emp_count"),
        avg("basicpay").alias("avg_salary"),
        max("basicpay").alias("max_salary"),
        sum("basicpay").alias("total_salary")
    ).withColumn(
        "budget_utilization", 
        col("total_salary") / col("budget") * 100
    )
```

### Pivot and Unpivot Operations
```python
# PySpark: Pivot
pivot_df = df.groupBy("deptcode") \
    .pivot("sex") \
    .agg(count("empcode"), avg("basicpay"))

# Pandas: Pivot
pivot_df = df.pivot_table(
    index='deptcode', 
    columns='sex', 
    values='basicpay', 
    aggfunc=['count', 'mean']
)
```

This cheat sheet covers the essential patterns you'll need for complex data analysis tasks similar to your SQL lab practice questions. The key is understanding how these operations translate between SQL, PySpark, and Pandas, allowing you to choose the right tool for your specific use case.