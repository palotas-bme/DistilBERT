import bs4
import requests
import wikipediaapi
import torch
from transformers import AutoTokenizer, DistilBertForQuestionAnswering
from utils import removeduplicates


class ContextFetcher():
    # Function for turning the question into a search term
    def create_search_term(self, question):
        pass # Placeholder

    # Function for searching for the extracted words from the question on google
    @removeduplicates(checkvals=["link"])
    def google_search(self, query, num_results=10):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        params = {
            "q": query,
            "num": num_results,
            "hl": "en",
        }
        url = "https://www.google.com/search"
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch search results: {response.status_code}")
        
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        results = []
        for g in soup.find_all("div", class_="tF2Cxc"):
            title = g.find("h3")
            link = g.find("a", href=True)
            if title and link:
                results.append({
                    "title": title.text,
                    "link": link["href"]
                })

        return results

    # Function for extracting context from wikipedia pages
    @removeduplicates(checkvals=["text"])
    def get_wiki_contexts(self, results):
        wiki_wiki = wikipediaapi.Wikipedia("Questionansweringproject-trial1", "en")
        contexts = []
        for result in results:
            title = result["title"].strip(" - Wikipedia").replace(" ", "_")
            page = wiki_wiki.page(title)
            if page.exists():
                contexts.append({
                    "text" : page.summary,
                    "link": page.fullurl
                })

        return results
    
    # Complete context fetching workflow
    def fetch_context(self, question):
        search_term = self.create_search_term(question)
        search_results = self.google_search(search_term)
        wiki_contexts = self.get_wiki_contexts(search_results)
        return wiki_contexts


class QuestionAnswerer():
    def __init__(self, tokenizer_path="distilbert/distilbert-base-uncased-distilled-squad", 
                 model_path="distilbert/distilbert-base-uncased-distilled-squad"):
        self.load_tokenizer(tokenizer_path)
        self.load_model(model_path)
        self.context_fetcher = ContextFetcher()

    # Answering a question
    def answer_question(self, question, context):
        if context:
            return self._answer_context(question, context)

        else:
            return self._answer_nocontext(question)
        
    # Answering a question when there is a context provided
    def _answer_context(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start_index = torch.argmax(outputs.start_logits)
        answer_end_index = torch.argmax(outputs.end_logits)

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index]
        answer = self.tokenizer.decode(predict_answer_tokens)

        return answer
    
    # Answering a question when no context is provided
    @removeduplicates(checkvals=["answer"])
    def _answer_nocontext(self, question):
        results = self._get_context(question)
        for result in results:
            result["answer"] = self._answer_context(question, result["text"])
        
        return results
    
    # Retrieving context to a question
    def _get_context(self, question):
        return self.context_fetcher.fetch_context(question)

    # Loading preferred pretrained tokenizer
    def load_tokenizer(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    # Loading preferred pretrained model 
    def load_model(self, path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DistilBertForQuestionAnswering.from_pretrained(path).to(device)

    # Moving model to different device if neccesary
    def change_device(self, device):
        self.model.to(device)
