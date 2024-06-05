import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from transformers import GPT2TokenizerFast
import ollama
import json
import requests
from bs4 import BeautifulSoup
import urllib.parse
import scrapy
from scrapy.crawler import CrawlerProcess

question = "search for rowtonsoftware chatcompletion."
searched_question = "what is the context in this page about? if there is an article, explain what its about? if there is product specifications, list them.:"

# Initialize the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Initialize the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Configuration for the Ollama API client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)

# question = "solve for `x` in 3x + 11 = 14?"
schema = {
    "is_search_request": {
        "type": "boolean",
        "description": "Whether the prompt is a request to perform search on a topic or item.",
    },
    "search_query": {
        "type": "string",
        "description": "The search query for the research request. This field should be empty if the prompt is not a research request.",
    },
    "response": {
        "type": "string",
        "description": "The response to the prompt. this field should be blank if prompt contains a research question.",
    }
}
prompt = f"RESPOND ONLY IN THIS JSON FORMAT USING THIS SCHEMA: {schema}\n\n {question}"
response = ollama.generate('llama3', prompt)
jsonVal = json.loads(response["response"])

def google_search(query, num_results=10):
    query = urllib.parse.quote_plus(query)
    url = f"https://www.google.com/search?q={query}&num={num_results}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = []

    for item in soup.find_all('div', class_='tF2Cxc'):
        link = item.find('a', href=True)
        if link:
            search_results.append(link['href'])

    return search_results

class MySpider(scrapy.Spider):
    name = "quotes"
    crawled_data = []

    def start_requests(self):
        print ("Searching Google For: " + query)
        urls = google_search(query)
        print ("Found: " + str(len(urls)) + " results")
        print ("Search Result URLs: " + str(urls))
        yield scrapy.Request(url=urls[0], callback=self.parse)

    def parse(self, response):
        print("Adding response to crawled data: " + response.url)
        print("Response Data:\n\n " + response.text + "\n\n")
        MySpider.crawled_data.append(response.text)

def start_crawl():
    print("Starting crawl...")

    process = CrawlerProcess(settings={
        'FEEDS': {
            'result.json': {'format': 'json'},
        },
        'LOG_LEVEL': 'WARNING',
        'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7'
    })

    process.crawl(MySpider)
    process.start()  # the script will block here until the crawling is finished

    # Join the crawled data into a single string and return it
    return ' '.join(MySpider.crawled_data)

def tokenize_and_summarize(text, max_tokens=1024):
    print("Tokenizing and summarizing text...")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    summaries = []

    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, clean_up_tokenization_spaces=True)

    return chunk_text

def embed_and_print(text):
    # Encode the text
    embeddings = model.encode([text])
    # Convert to tensor and print embeddings
    embeddings_tensor = torch.tensor(embeddings)
    return text, embeddings_tensor

def get_relevant_context(user_input, vault_embeddings, vault_content, model, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the user input
    input_embedding = model.encode([user_input])
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0]
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

if jsonVal["is_search_request"]:
    MAX_LENGTH = 2048
    query = jsonVal["search_query"]
    print(f"Researching {query}...")
    result = start_crawl()
    result_text, tensor = embed_and_print(result)
    context = get_relevant_context(result_text, tensor, [result_text], model)
    if context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(context)
        # Truncate the context string if it's too long
        if len(context_str) > MAX_LENGTH:
            context_str = context_str[:MAX_LENGTH]
            
        resp = ollama.generate('llama3', f"{searched_question}\n\n" + str(context_str))
        print(resp["response"])
