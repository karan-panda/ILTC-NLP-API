import pandas as pd
import psycopg2

# Database connection
conn = psycopg2.connect(
    dbname="iltc_db",
    user="postgres",
    password="admin",
    host="localhost",
    port="5432"
)

query = "SELECT * FROM user_queries"
df = pd.read_sql(query, conn)
df.to_csv("data.csv", index=False)

conn.close()
print("Data unloaded successfully to data.csv!")