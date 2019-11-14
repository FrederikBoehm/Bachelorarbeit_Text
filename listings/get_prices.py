sql_date = f'{year}-{month}-{day}'

sql_command = f"""
SELECT open, close FROM trading_day
WHERE date >= '{sql_date}'
AND ticker = '{ticker}'
ORDER BY date ASC
LIMIT 3;"""

db_cursor.execute(sql_command)
prices = db_cursor.fetchall()