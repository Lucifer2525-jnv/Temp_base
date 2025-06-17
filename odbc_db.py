import pyodbc
import os
from azure.identity import ClientSecretCredential

# Your existing variables
odbc_driver_name = "{ODBC Driver 17 for SQL Server}"
server = config['server']
database = "CBLearning"
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
tenant_id = os.getenv('tenant_id')  # You'll need this too

try:
    # Method 1: Using Azure Identity for Access Token (Recommended for Azure SQL)
    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret
    )
    
    # Get access token for Azure SQL Database
    token = credential.get_token("https://database.windows.net/.default")
    
    # Create connection string with access token
    connString = (
        f"Driver={odbc_driver_name};"
        f"Server={server};"
        f"Database={database};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )
    
    # Create connection with access token
    conn = pyodbc.connect(connString, attrs_before={
        1256: token.token.encode('utf-16-le')  # SQL_COPT_SS_ACCESS_TOKEN
    })
    
    print("Connected successfully using access token!")
    
except Exception as e:
    print(f"Method 1 failed: {e}")
    
    try:
        # Method 2: Direct Authentication (if supported by your SQL Server setup)
        connString = (
            f"Driver={odbc_driver_name};"
            f"Server={server};"
            f"Database={database};"
            f"Authentication=ActiveDirectoryServicePrincipal;"
            f"UID={client_id};"
            f"PWD={client_secret};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout=30;"
        )
        
        conn = pyodbc.connect(connString)
        print("Connected successfully using service principal!")
        
    except Exception as e2:
        print(f"Method 2 failed: {e2}")
        
        # Method 3: If you have SQL Server authentication credentials
        # This would require different username/password, not client_id/secret
        sql_username = os.getenv('sql_username')  # Different from client_id
        sql_password = os.getenv('sql_password')  # Different from client_secret
        
        if sql_username and sql_password:
            try:
                connString = (
                    f"Driver={odbc_driver_name};"
                    f"Server={server};"
                    f"Database={database};"
                    f"UID={sql_username};"
                    f"PWD={sql_password};"
                    f"Encrypt=yes;"
                    f"TrustServerCertificate=no;"
                    f"Connection Timeout=30;"
                )
                
                conn = pyodbc.connect(connString)
                print("Connected successfully using SQL authentication!")
                
            except Exception as e3:
                print(f"Method 3 failed: {e3}")
        else:
            print("No SQL authentication credentials available")

# Don't forget to close the connection when done
# conn.close()