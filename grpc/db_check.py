import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('myapp.db')

# Create a cursor object to execute SQL queries
cursor = conn.cursor()

# Execute a SELECT query to retrieve all rows from the 'predictions' table
cursor.execute('SELECT * FROM predictions')

# Fetch all rows from the cursor
rows = cursor.fetchall()

# Print the rows
for row in rows:
    print(row)

# Close the cursor and connection
cursor.close()
conn.close()
