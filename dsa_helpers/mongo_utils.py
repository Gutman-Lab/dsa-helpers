import json
from os import getenv
from gridfs import GridFS
import numpy as np
from io import BytesIO
from bson.objectid import ObjectId
from PIL import Image
from typing import Union

import pymongo
from pymongo.database import Database


def store_json_in_db(mongo_db, json_data: dict, _id: str | None = None) -> str:
    """Store a dictionary as a JSON file in GridFS.

    Args:
        mongo_db: The MongoDB database.
        json_data (dict): The dictionary to store.

    Returns:
        str: The ID of the stored JSON file.
    """
    fs = GridFS(mongo_db)

    # Convert dictionary to JSON string
    json_str = json.dumps(json_data)

    # Store the JSON data in GridFS
    if _id is not None:
        file_id = fs.put(
            json_str.encode("utf-8"),
            encoding="utf-8",
            content_type="application/json",
            _id=ObjectId(_id),
        )
    else:
        file_id = fs.put(
            json_str.encode("utf-8"),
            encoding="utf-8",
            content_type="application/json",
        )

    return str(file_id)


def get_mongo_client(
    protocol: str | None = None,
    host: str | None = None,
    username: str | None = None,
    password: str | None = None,
    port: int | None = None,
):
    """Get a mongo client by providing the key parts of the URL. By default
    when the parameters are not passed, it sets them to default values first
    by looking at key environment variables. If the environment variables are
    not present it sets them to default values.

    Args:
        protocol (str, optional): The protocol of the mongo server. If None it
            look for the environment variable "MONGO_PROTOCOL". If this is
            missing it will default to "mongodb".
        host (str, optional): The host name of the mongo server. If None it will
            look for the environment variable "MONGO_HOST_NAME". If this is
            missing it will default to "mongodb".
        username (str, optional): The username to connect to the mongo server.
            If None it will look for the environment variable
            "MONGO_INITDB_ROOT_USERNAME". If this is missing it will default to
            "docker".
        password (str, optional): The password to connect to the mongo server.
            If None it will look for the environment variable
            "MONGO_INITDB_ROOT_PASSWORD". If this is missing it will default to
            "docker".
        port (int, optional): The port number of the mongo server. If None it
            will look for the environment variable "MONGO_HOST_PORT". If this is
            missing it will default to 27017.

    Returns:
        The mongo client.

    """
    if protocol is None:
        protocol = getenv("MONGO_PROTOCOL", "mongodb")
    if username is None:
        username = getenv("MONGO_INITDB_ROOT_USERNAME", "docker")
    if password is None:
        password = getenv("MONGO_INITDB_ROOT_PASSWORD", "docker")
    if host is None:
        host = getenv("MONGO_HOST_NAME", "mongodb")
    if port is None:
        port = int(getenv("MONGO_HOST_PORT", 27017))

    return pymongo.MongoClient(
        f"{protocol}://{username}:{password}@{host}:{port}"
    )


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
    mongo_collection,
    items: dict | list,
    key: str = "_id",
) -> dict[str | dict]:
    """Add items to a mongo collection. For this project we always add items with a unique
    user key.

    Args:
        mongo_collection: The mongo collection to add to.
        items (dict | list): The items to add to the collection.
        key (str): The key to use to identify the items.

    Returns:
        dict[str | dict]: The items added to the collection.

    """
    # If the items are a list, then use the "key" parameter to nest it as a dict.
    if isinstance(items, list):
        items = {item[key]: item for item in items}

    operations = []

    for _id, item in items.items():
        operations.append(
            pymongo.UpdateOne({"_id": _id}, {"$set": item}, upsert=True)
        )

    for chunk in chunks(operations):
        _ = mongo_collection.bulk_write(chunk)

    return items


def get_img_from_db(mongo_db, img_id: str) -> Union[np.ndarray, None]:
    """Get an image from the database by its location id.

    Args:
        mongo_db: The mongo database.
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
    mongo_db,
    img: Image.Image | np.ndarray,
) -> str:
    """Add an image to the database.

    Args:
        mongo_db: The mongo database.
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


def store_json_in_gridfs(
    mongo_db: Database, json_data: dict, _id: str | None = None
) -> str:
    """Store a dictionary as a JSON file in GridFS.

    Args:
        mongo_db (pymongo.database.Database): The MongoDB database.
        json_data (dict): The dictionary to store.
        _id (str, optional): The ID to use for the stored JSON file. If None,
            a new ID will be generated.

    Returns:
        str: The ID of the stored JSON file.
    """
    fs = GridFS(mongo_db)

    # Convert dictionary to JSON string
    json_str = json.dumps(json_data)

    # Store the JSON data in GridFS
    if _id is not None:
        file_id = fs.put(
            json_str.encode("utf-8"),
            encoding="utf-8",
            content_type="application/json",
            _id=ObjectId(_id),
        )
    else:
        file_id = fs.put(
            json_str.encode("utf-8"),
            encoding="utf-8",
            content_type="application/json",
        )

    return str(file_id)


def get_json_from_gridfs(mongo_db: Database, json_id: str) -> dict:
    """Retrieve a JSON dictionary from GridFS.

    Args:
        mongo_db (pymongo.database.Database): The MongoDB database.
        json_id (str): The GridFS file ID.

    Returns:
        dict: The retrieved dictionary.

    """
    fs = GridFS(mongo_db)

    grid_out = fs.get(ObjectId(json_id))

    if grid_out:
        # Read the JSON string and convert it back to a dictionary
        json_str = grid_out.read().decode("utf-8")
        return json.loads(json_str)

    return None  # If not found
