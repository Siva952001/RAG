import pymongo
from pymongo import MongoClient

def get_mongo_client():
    # Replace with your MongoDB connection details
    return MongoClient("mongodb://localhost:27017/")

def get_database(client):
    return client['RAG-STORE']

def get_collection(db):
    return db['llama']

def save_chat_to_db(collection, chat_data):
    collection.insert_one(chat_data)
