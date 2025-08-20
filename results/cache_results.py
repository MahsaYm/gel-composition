import sqlite3, functools, hashlib, time, re, pickle, base64
from pathlib import Path
from ast import literal_eval

# Fixed database path inside the results folder next to this file
DB_PATH = Path(__file__).resolve().parent / 'cache_results.sqlite'

def disk_cache():
	"""Persistent cache.
	- DB always at results/cache_results.sqlite
	- Table name always the function name (sanitized)
	- Stores args, kwargs, result as repr strings
	"""
	DB_PATH.parent.mkdir(parents=True, exist_ok=True)
	def deco(func):
		# Sanitize table: only letters, digits, underscore & not start with digit
		raw_table = func.__name__
		table = re.sub(r'\W', '_', raw_table)
		if table[0].isdigit():
			table = f't_{table}'
		qual = f"{func.__module__}.{func.__name__}"
		def ensure_table():
			with sqlite3.connect(DB_PATH) as conn:
				conn.execute(
					f"CREATE TABLE IF NOT EXISTS \"{table}\" (key TEXT PRIMARY KEY, args TEXT, kwargs TEXT, result TEXT, created REAL)"
				)
		# Initial attempt (best effort)
		ensure_table()

		@functools.wraps(func)
		def wrapper(*a, **kw):
			args_s = repr(a); kwargs_s = repr(kw)
			key = hashlib.sha1(f"{qual}|{args_s}|{kwargs_s}".encode()).hexdigest()
			with sqlite3.connect(DB_PATH) as conn:
				try:
					cur = conn.execute(f"SELECT result FROM \"{table}\" WHERE key=?", (key,))
					row = cur.fetchone()
				except sqlite3.OperationalError as e:
					# Table might not exist (e.g., module reloaded in notebook); create then retry once
					if 'no such table' in str(e).lower():
						ensure_table()
						cur = conn.execute(f"SELECT result FROM \"{table}\" WHERE key=?", (key,))
						row = cur.fetchone()
					else:
						raise
				if row:
					val_s = row[0]
					try:
						# Try pickle deserialization first (for complex objects)
						if val_s.startswith('PICKLE:'):
							return pickle.loads(base64.b64decode(val_s[7:]))
						# Fall back to literal_eval for simple objects
						return literal_eval(val_s)
					except Exception:
						return val_s
			res = func(*a, **kw)
			with sqlite3.connect(DB_PATH) as conn:
				try:
					# Try to serialize with repr first (for simple objects)
					try:
						res_str = repr(res)
						# Test if it can be deserialized
						literal_eval(res_str)
					except (ValueError, SyntaxError):
						# Fall back to pickle for complex objects
						res_str = 'PICKLE:' + base64.b64encode(pickle.dumps(res)).decode('ascii')
					
					conn.execute(f"INSERT OR REPLACE INTO \"{table}\" (key,args,kwargs,result,created) VALUES (?,?,?,?,?)", (key, args_s, kwargs_s, res_str, time.time()))
				except sqlite3.OperationalError as e:
					if 'no such table' in str(e).lower():
						ensure_table()
						conn.execute(f"INSERT OR REPLACE INTO \"{table}\" (key,args,kwargs,result,created) VALUES (?,?,?,?,?)", (key, args_s, kwargs_s, res_str, time.time()))
					else:
						raise
			return res
		wrapper.cache_db = str(DB_PATH)  # type: ignore[attr-defined]
		wrapper.cache_table = table      # type: ignore[attr-defined]
		return wrapper
	return deco




if __name__ == '__main__':
	# Example usage
	from time import sleep

	@disk_cache()
	def expensive_computation(x):
		sleep(0.2)  # Simulate a delay
		return x * x  # Simulate a heavy computation

	print(expensive_computation(10))  # First call, computes and caches
	print(expensive_computation(10))  # Second call, retrieves from cache
	print(expensive_computation(20))  # Computes and caches new value

	@disk_cache()
	def add(x, y):
		print('computing add')
		sleep(0.1)  # Simulate a delay
		return x + y
	print(add(2,3))  # compute
	print(add(2,3))  # cached