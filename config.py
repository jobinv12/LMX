from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 1.0,
    max_tokens = None,
    timeout = 10,
    max_retries=1
)

HF_BUCKET_NAME = os.environ.get("HF_BUCKET_NAME")
HF_BUCKET_URL = os.environ.get("HF_BUCKET_URL")
HF_BUCKET_TOKEN = os.environ.get("HF_BUCKET_TOKEN")