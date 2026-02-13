from typing import Optional, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo.errors import ConfigurationError
from config import MONGO_URI, MONGO_DB_NAME, MONGO_USERS_COLLECTION, logger

mongo_client: Optional[AsyncIOMotorClient] = None
users_collection: Optional[AsyncIOMotorCollection] = None
database: Optional[Any] = None

if MONGO_URI:
    try:
        mongo_client = AsyncIOMotorClient(MONGO_URI)
        try:
            database = mongo_client.get_default_database()
        except ConfigurationError:
            if MONGO_DB_NAME:
                database = mongo_client[MONGO_DB_NAME]
            else:
                database = None
                logger.warning(
                    "Mongo URI does not include a database name and MONGO_DB_NAME is not set."
                )
        if database is not None:
            users_collection = database[MONGO_USERS_COLLECTION]
            logger.info(
                "MongoDB connected for auth features (collection: %s)",
                MONGO_USERS_COLLECTION,
            )
    except Exception as exc:
        logger.error("MongoDB connection error: %s", exc)
else:
    logger.warning("MONGO_URI not provided. Auth endpoints will be unavailable.")

def get_users_collection() -> AsyncIOMotorCollection:
    return users_collection
