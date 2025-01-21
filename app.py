
from fastapi import FastAPI
from langchain_ollama import OllamaLLM
# depreciated 
# from langchain_community.llms import Ollama

app = FastAPI(__name__)

llm = OllamaLLM(model = "llama3")

response = llm.invoke("Tell me a cat joke")

print(response)