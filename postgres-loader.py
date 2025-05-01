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

cursor = conn.cursor()

df = pd.read_csv("Training/cleaned_train_data.csv")

for index, row in df.iterrows():
    cursor.execute("INSERT INTO user_queries (user_input, user_intent, score) VALUES (%s, %s, %s)",
                   (row['user_input'], row['user_intent'], row['score']))

conn.commit()
cursor.close()
conn.close()

print(f"Data loaded successfully into {conn.info.dbname} database.")