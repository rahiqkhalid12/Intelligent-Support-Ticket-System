import sqlite3

conn = sqlite3.connect("support.db")  # <-- replace with your actual filename
cur = conn.cursor()
cur.execute("ALTER TABLE tickets ADD COLUMN updated_at DATETIME")
cur.execute("UPDATE tickets SET updated_at = created_at WHERE updated_at IS NULL")
conn.commit()
conn.close()
print("Migration complete.")