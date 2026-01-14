from fastapi import FastAPI, HTTPException
import psycopg2

app = FastAPI()

def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="testdb",
        user="postgres",
        password="postgres"
    )
    return conn

@app.get("/")
def root():
    return {"message": "Qureka Backend + DB 연결 테스트 중!"}

@app.get("/db-test")
def db_test():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, name TEXT UNIQUE);")

        users = [("Alice",), ("Bob",), ("dummy user",)]
        for user in users:
            cur.execute("INSERT INTO users (name) VALUES (%s) ON CONFLICT (name) DO NOTHING;", user)

        conn.commit()

        cur.execute("SELECT id, name FROM users;")
        rows = cur.fetchall()

        cur.close()
        conn.close()

        return {"data": [{"id": r[0], "name": r[1]} for r in rows]}

    except Exception as e:
        print("DB ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=repr(e))
