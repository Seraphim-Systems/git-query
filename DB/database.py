from DB.db_config import db_clients, get_mongodb_db, get_cosmos_db, get_qdrant_client

class DatabaseManager:
    def __init__(self):
        self._clients = db_clients
    def get_mongodb(self):
        return get_mongodb_db()
    def get_cosmos(self):
        return get_cosmos_db()
    def get_qdrant(self):
        return get_qdrant_client()
    def close_all(self):
        self._clients.close_all()
    @property
    def config(self):
        return self._clients.config

db_manager = DatabaseManager()
__all__ = ['DatabaseManager', 'db_manager', 'get_mongodb_db', 'get_cosmos_db', 'get_qdrant_client']
