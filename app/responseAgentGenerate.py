from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"

load_dotenv(find_dotenv())
client = OpenAI()


