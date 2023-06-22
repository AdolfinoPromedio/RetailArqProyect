from pymongo import MongoClient
from datetime import datetime

# Establish connection with MongoDB
client = MongoClient('mongodb://localhost:27017')  # Change this according to your configuration

# Create a new database
database_name = 'ProjectX'
db = client[database_name]

# Create a collection for the images
collection_name = 'images'
collection = db[collection_name]

def mongo_save_image(image_path):
    # Read the image as binary
    with open(image_path, 'rb') as file:
        image_bytes = file.read()

    # Get the current date and time
    current_datetime = datetime.now()

    # Create a document with the image metadata
    document = {
        'image': image_bytes,
        'upload_date': current_datetime
    }

    # Insert the document into the collection
    collection.insert_one(document)



