from pymongo import MongoClient

def get_mongo_client(uri="mongodb://localhost:27017/"):
    return MongoClient(uri)

def get_database(client, db_name="RAG-STORE"):
    return client[db_name]

def get_collection(db, collection_name="llama"):
    return db[collection_name]

def save_chat_to_db(collection, chat_data):
    collection.insert_one(chat_data)
