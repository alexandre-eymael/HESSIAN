"""
Database module for handling database operations.
This module provides a class to interact with the database.
"""

import sqlite3

class HessianDatabase:
    """
    Class for the SQLite database.
    """

    def __init__(self, db_file):
        """
        Initialize the database.

        Args:
            db_file (str): The path to the database file.
        """

        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.cursor = self.conn.cursor()

    def load_schema(self, schema):
        """
        Load the schema into the database.

        Args:
            schema (str): The path to the SQL schema file.
        """
        with open(schema, "r", encoding='utf-8') as f:
            self.cursor.executescript(f.read())
            self.conn.commit()

    def load_data(self, data):
        """
        Load the data into the database.

        Args:
            data (str): The path to the SQL data file.
        
        Returns:

        """
        with open(data, "r", encoding='utf-8') as f:
            self.cursor.executescript(f.read())
            self.conn.commit()

    def init_if_empty(self, schema, data):
        """
        Initialize the database with the schema and data if it is empty.

        Args:
            schema (str): The path to the SQL schema file.
            data (str): The path to the SQL data file.
        """
        self.cursor.execute("SELECT * FROM sqlite_master WHERE type='table';")
        if not self.cursor.fetchall():
            self.load_schema(schema)
            self.load_data(data)

    def get_user_from_api_key(self, api_key):
        """
        Get the user from the API key.

        Args:
            api_key (str): The API key.

        Returns:
            str: The user associated with the API key.
        """
        self.cursor.execute("SELECT * FROM users WHERE api_key = ?", (api_key,))
        user = self.cursor.fetchone()
        return user if user else None

    def get_models(self):
        """
        Get the models from the database.

        Returns:
            list: The models in the database.
        """
        self.cursor.execute("SELECT * FROM models")
        return self.cursor.fetchall()

    def get_model_id_by_name(self, model_name):
        """
        Get the model ID by the model name.

        Args:
            model_name (str): The model name.

        Returns:
            int: The model ID.
        """
        self.cursor.execute("SELECT model_id FROM models WHERE model_name = ?", (model_name,))
        item = self.cursor.fetchone()
        return str(item[0]) if item else None


    def add_query(self, user_id, model_id):
        """
        Add a query to the database.

        Args:
            user_id (int): The user ID.
            model_id (int): The model ID.
        """
        self.cursor.execute("INSERT INTO queries (user_id, model_id) VALUES (?, ?)",
                            (user_id, model_id))
        self.conn.commit()

    def get_all_queries(self):
        """
        Get all the queries from the database.

        Returns:
            list: The queries in the database.
        """
        query = """
        SELECT q.query_id, q.creation_date, m.model_name
        FROM queries q
        JOIN models m ON q.model_id = m.model_id
        ORDER BY q.creation_date
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_queries(self, api_key):
        """
        Get the queries linked to the user whose API key is `api_key`.

        Args:
            api_key (str): The API key of the user.

        Returns:
            list: The queries linked to the user.
        """
        query = """
        SELECT q.query_id, q.user_id, q.model_id, m.model_price, m.model_name
        FROM queries q
        JOIN users u ON q.user_id = u.user_id
        JOIN models m ON q.model_id = m.model_id
        WHERE u.api_key = ?
        """
        self.cursor.execute(query, (api_key,))
        return self.cursor.fetchall()
