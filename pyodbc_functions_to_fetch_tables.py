import pyodbc

# Assuming you already have your working connection
# conn = pyodbc.connect(connString)

def list_tables_method1(conn):
    """Method 1: Using INFORMATION_SCHEMA.TABLES (Standard SQL)"""
    cursor = conn.cursor()
    query = """
    SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'BASE TABLE'
    ORDER BY TABLE_SCHEMA, TABLE_NAME
    """
    
    cursor.execute(query)
    tables = cursor.fetchall()
    
    print("=== Method 1: INFORMATION_SCHEMA.TABLES ===")
    print(f"Found {len(tables)} tables:")
    for table in tables:
        print(f"  {table.TABLE_SCHEMA}.{table.TABLE_NAME}")
    print()
    
    return tables

def list_tables_method2(conn):
    """Method 2: Using sys.tables (SQL Server specific)"""
    cursor = conn.cursor()
    query = """
    SELECT 
        s.name AS schema_name,
        t.name AS table_name,
        t.create_date,
        t.modify_date
    FROM sys.tables t
    INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
    ORDER BY s.name, t.name
    """
    
    cursor.execute(query)
    tables = cursor.fetchall()
    
    print("=== Method 2: sys.tables (with dates) ===")
    print(f"Found {len(tables)} tables:")
    for table in tables:
        print(f"  {table.schema_name}.{table.table_name} (Created: {table.create_date}, Modified: {table.modify_date})")
    print()
    
    return tables

def list_tables_method3(conn):
    """Method 3: Simple table names only"""
    cursor = conn.cursor()
    query = """
    SELECT name 
    FROM sys.tables 
    ORDER BY name
    """
    
    cursor.execute(query)
    tables = cursor.fetchall()
    
    print("=== Method 3: Simple table names ===")
    print(f"Found {len(tables)} tables:")
    for table in tables:
        print(f"  {table.name}")
    print()
    
    return tables

def get_table_info(conn, schema_name='dbo', table_name=None):
    """Get detailed information about columns in a specific table"""
    cursor = conn.cursor()
    
    if table_name:
        query = """
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION
        """
        cursor.execute(query, schema_name, table_name)
    else:
        print("Please provide a table name")
        return
    
    columns = cursor.fetchall()
    
    print(f"=== Columns in {schema_name}.{table_name} ===")
    for col in columns:
        nullable = "NULL" if col.IS_NULLABLE == "YES" else "NOT NULL"
        max_length = f"({col.CHARACTER_MAXIMUM_LENGTH})" if col.CHARACTER_MAXIMUM_LENGTH else ""
        default = f" DEFAULT {col.COLUMN_DEFAULT}" if col.COLUMN_DEFAULT else ""
        print(f"  {col.COLUMN_NAME}: {col.DATA_TYPE}{max_length} {nullable}{default}")

# Usage example:
try:
    # List tables using different methods
    tables1 = list_tables_method1(conn)
    tables2 = list_tables_method2(conn)
    tables3 = list_tables_method3(conn)
    
    # Get details of a specific table (replace 'your_table_name' with actual table name)
    # get_table_info(conn, 'dbo', 'your_table_name')
    
except Exception as e:
    print(f"Error: {e}")

# Alternative: Quick one-liner to just get table names
def quick_table_list(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sys.tables ORDER BY name")
    return [row.name for row in cursor.fetchall()]

# Usage:
# table_names = quick_table_list(conn)
# print("Tables:", table_names)