import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


class LLM:
    def __init__(self, model_name:str):
        self.model_name = model_name

        match model_name:
            case "ministral/Ministral-3b-instruct":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map= "cuda" if torch.cuda.is_available() else "cpu"
                )
            case _:
                raise ValueError(f"Unsupported model name: {self.model_name}")

    def generate(self,
                messages:list[dict],
                max_new_tokens:int=200,
                temperature:float=0.7,
                top_p:float=0.9,
        ) -> str:

        match self.model_name:
            case "ministral/Ministral-3b-instruct":
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
                    )
                
                response = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                )
                return response 
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
