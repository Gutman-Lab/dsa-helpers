import pymongo
from os import getenv
from gridfs import GridFS
import numpy as np
from io import BytesIO
from bson.objectid import ObjectId
from PIL import Image
from typing import Union


def get_mongo_database(mongo_url: str | None = None, database: str | None = None) -> pymongo.database.Database:
    """Get a mongo database. By default the Mongo URL will be assumed from local
    environment variables:
        "mongodb://$MONGO_INITDB_ROOT_USERNAME:$MONGO_INITDB_ROOT_PASSWORD@mongodb:27017"
        
    and the database will be assumed from the environment variable "MONGODB_DB".
    
    Args:
        mongo_url (str): The mongo URL.
        database (str): The database name.
        
    Returns:
        pymongo.database.Database: The mongo database.
    
    """
    if mongo_url is None:
        mongo_url = f"mongodb://{getenv('MONGO_INITDB_ROOT_USERNAME')}:" + \
                    f"{getenv('MONGO_INITDB_ROOT_PASSWORD')}" + \
                    f"@{getenv('MONGO_HOST_NAME')}:{getenv("MONGO_HOST_PORT")}"
    
    mc = pymongo.MongoClient(mongo_url)
    
    if database is None:
        database = getenv("MONGODB_DB")

    # Return the specific database.
    return mc[database]


def chunks(lst: list, n=500):
    """Helper function for traversting through a list in chunks.

    Args:
        lst (list): The list to traverse.
        n (int): The size of the chunks.

    Returns:
        generator: A generator of the list in chunks.

    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
        
        
def add_many_to_collection(
    mongo_collection: pymongo.collection.Collection,
    items: dict | list, key: str = "_id"
) -> dict[str | dict]:
    """Add items to a mongo collection. For this project we always add items with a unique
    user key.
    
    Args:
        mongo_collection (pymongo.collection.Collection): The mongo collection to add to.
        items (dict | list): The items to add to the collection.
        key (str): The key to use to identify the items.
        
    Returns:
        dict[str | dict]: The items added to the collection.

    """
    # If the items are a list, then use the "key" parameter to nest it as a dict.
    if isinstance(items, list):
        items = {item[key]: item for item in items}

    operations = []

    for item in items:
        operations.append(
            pymongo.UpdateOne({"_id": item[key]}, {"$set": item}, upsert=True)
        )

    for chunk in chunks(operations):
        _ = mongo_collection.bulk_write(chunk)

    return items


def get_img_from_db(mongo_db: pymongo.database.Database, img_id: str) -> Union[np.ndarray, None]:
    """Get an image from the database by its location id.

    Args:
        mongo_db (pymongo.database.Database): The mongo database.
        img_loc (str): The image id in mongo.

    Returns:
        np.array: The image, if None then the image was not found.

    """
    fs = GridFS(mongo_db)

    grid_out = fs.get(ObjectId(img_id))

    if grid_out:
        # Read the byte data from the GridOut object
        byte_data = grid_out.read()

        # Create a BytesIO object from the byte data
        byte_io = BytesIO(byte_data)

        # Open the image from the BytesIO object
        src_img = np.array(Image.open(byte_io))

        return src_img
    
    
def add_img_to_db(
    mongo_db: pymongo.database.Database,
    img: Image.Image | np.ndarray,
) -> str:
    """Add an image to the database.

    Args:
        mongo_db (pymongo.database.Database): The mongo database.
        img (np.ndarray or PIL.Image.Image): The image.
        
    Returns:
        str: The image id in the database.

    """
    fs = GridFS(mongo_db)

    # Convert image to PIL image if needed.
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Create a BytesIO object and save the image to it
    byte_io = BytesIO()
    img.save(byte_io, "PNG")

    # Get the byte data from the BytesIO object
    byte_io.seek(0)
    byte_data = byte_io.read()

    # Save the byte data to MongoDB using GridFS
    file_id = fs.put(byte_data)

    return file_id