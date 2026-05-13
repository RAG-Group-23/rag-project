import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer
from prompts import MAIN_ASSISTANT_PROMPT
from transformers import BitsAndBytesConfig
import os 

class LLM:
    def __init__(self, model_name:str, load_model=True):
        self.model_name = model_name
        self.root = os.getenv("MODEL_ROOT", "/files")
        match model_name:
            case "ministral/Ministral-3b-instruct":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if load_model:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="cuda" if torch.cuda.is_available() else "cpu"
                    )
            case "google/gemma-3-4b-it":
                model_name_or_path = f"{self.root}/google-gemma-3-4b-it" if os.path.exists(f"{self.root}/google-gemma-3-4b-it") else "google/gemma-3-4b-it"
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=torch.float16,
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
                ).eval()
                self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            case _:
                raise ValueError(f"Unsupported model name: {self.model_name}")

    def generate(self,
                messages:list[dict],
                documents:list[str],
                max_new_tokens:int=200,
                temperature:float=0.7,
                top_p:float=0.9,
        ) -> str:

        match self.model_name:
            #------------------------------------------------------------ 
            #-----ministral/Ministral-3b-instruct------------------------ 
            #------------------------------------------------------------ 
            case "ministral/Ministral-3b-instruct":
                # print(f"Conversation: {messages}")

                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        repetition_penalty=1.1,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                response = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                )
                return response 

            #------------------------------------------------------------ 
            #-----google/gemma-3-4b-it----------------------------------- 
            #------------------------------------------------------------ 
            case "google/gemma-3-4b-it":

                inputs = self.format_prompt(messages, documents) 

                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                    )
                    generation = generation[0][input_len:]
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                return decoded
            #------------------------------------------------------------ 
            #------------------------------------------------------------ 
            #------------------------------------------------------------ 
            case _:
                raise ValueError(f"Unsupported model name: {self.model_name}")


    def format_prompt(self, conversation:list[dict], documents:list[str]):
        match self.model_name:

            #------------------------------------------------------------ 
            #------ministral/Ministral-3b-instruct----------------------- 
            #------------------------------------------------------------ 
            case  "ministral/Ministral-3b-instruct":
                conversation = [{"role":"system", "content": MAIN_ASSISTANT_PROMPT}] + conversation

                conversation[-1]["content"] += "\n\nRetrieved documents:"

                for i, doc in enumerate(documents):
                    conversation[-1]["content"] += f"\n[{i} document]\n {doc}\n"  

                if len(documents) == 0:
                    conversation[-1]["content"] += f"No relevant documents were found.\n"  

                final_prompt = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=True, 
                    add_generation_prompt=True,
                )
                return final_prompt


            #------------------------------------------------------------ 
            #-----google/gemma-3-4b-it----------------------------------- 
            #------------------------------------------------------------ 
            case "google/gemma-3-4b-it":
                # import json 
                # print(json.dumps(conversation, indent=2))
                for i, msg in enumerate(conversation):
                    conversation[i]["content"] = [{"type": "text", "text":msg["content"]}]

                conversation = [{"role":"system", "content": [{"type": "text", "text": MAIN_ASSISTANT_PROMPT}]}] + conversation


                conversation[-1]["content"][0]["text"] += "\n\nRetrieved documents:"

                for i, doc in enumerate(documents):
                    conversation[-1]["content"][0]["text"] += f"\n[{i} document]\n {doc}\n"  

                if len(documents) == 0:
                    conversation[-1]["content"][0]["text"] += f"No relevant documents were found.\n"  


                return self.processor.apply_chat_template(
                    conversation, 
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device, dtype=torch.float16)


            #------------------------------------------------------------ 
            #------------------------------------------------------------ 
            #------------------------------------------------------------ 
            case _:
                raise ValueError(f"Unsupported model name: {self.model_name}")


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name

        match self.model_name:

            # this model was chosen mostly at random. 
            # It has a hight score on the MTEB benchmark, and is relatively small, which should make it faster to run.
            case "Qwen/Qwen3-Embedding-4B":
                self.model = SentenceTransformer(
                    "Qwen/Qwen3-Embedding-4B",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )

            case _:
                raise ValueError(f"Unsupported model name: {self.model_name}")

    def encode(self, input: list[str], is_query: bool | None = None) -> list[list[float]]:

        if is_query is None:
            is_query = False

        match self.model_name:

            case "Qwen/Qwen3-Embedding-4B":
                return self.model.encode(
                    input,
                    prompt_name="query" if is_query else "document",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                ).tolist()

            case _:
                raise ValueError(f"Unsupported model name: {self.model_name}")


        
if __name__ == "__main__":
    llm = LLM("ministral/Ministral-3b-instruct")
    messages = [
        {"role": "user", "content": "Explain what attention is in a transformer model."}
    ]
    print("response: ")
    print(llm.generate(messages))


    embedding_model = Embedder("Qwen/Qwen3-Embedding-4B")
    
    texts =[
        "Some text", 
        "Some other text"
    ]

    embeddings = embedding_model.encode(texts, is_query=True)
    print(embeddings[0][:10])
    embeddings = embedding_model.encode(texts, is_query=False)
    print(embeddings[0][:10])

    while True:
        pass
