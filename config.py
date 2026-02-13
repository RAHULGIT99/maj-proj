import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Logger setup
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hey_backend")

# Env Vars
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in env")

API_KEY = os.getenv("API_KEY")

GROQ_API_KEYS = [k for k in (os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2"), os.getenv("GROQ_API_KEY_3")) if k]
if not GROQ_API_KEYS:
    raise ValueError("At least one GROQ_API_KEY_x environment variable must be set")

EMBEDDING_API_URL = "https://rahulbro123-embedding-model.hf.space/get_embeddings"
EMBEDDING_API_BATCH_SIZE = int(os.getenv("EMBEDDING_API_BATCH_SIZE", "32"))

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_USERS_COLLECTION = os.getenv("MONGO_USERS_COLLECTION", "users")

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
try:
    JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "1440"))
except ValueError:
    JWT_EXPIRATION_MINUTES = 1440

BREVO_API_KEY = os.getenv("BREVO_API_KEY")
BREVO_SENDER_EMAIL = "rahulvalavoju123@gmail.com"
BREVO_EMAIL_ENDPOINT = "https://api.brevo.com/v3/smtp/email"
