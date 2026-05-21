import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer
from prompts import MAIN_ASSISTANT_PROMPT
from transformers import BitsAndBytesConfig
from langchain_core.documents import Document
import os


class LLM:
    def __init__(self, model_name: str, load_model=True):
        self.model_name = model_name
        self.root = os.getenv("MODEL_ROOT", "/files")
        match model_name:
            case "Qwen/Qwen2.5-0.5B-Instruct":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if load_model:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,  # CPU-safe
                        device_map="cpu"
                    )
            case "ministral/Ministral-3b-instruct":
                model_name_or_path = f"{self.root}/Ministral-3b-instruct" if os.path.exists(
                    f"{self.root}/Ministral-3b-instruct") else "ministral/Ministral-3b-instruct"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if load_model:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="cuda" if torch.cuda.is_available() else "cpu"
                    )
            case "google/gemma-3-4b-it":
                model_name_or_path = f"{self.root}/google-gemma-3-4b-it" if os.path.exists(
                    f"{self.root}/google-gemma-3-4b-it") else "google/gemma-3-4b-it"
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=torch.bfloat16,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                ).eval()
                self.processor = AutoProcessor.from_pretrained(
                    model_name_or_path)


            case "HuggingFaceTB/SmolLM2-360M-Instruct":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if load_model:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        device_map="cpu"
                    )
            case _:
                raise ValueError(f"Unsupported model name: {self.model_name}")

    @staticmethod
    def _format_doc_header(i: int, doc: Document) -> str:
        """Build a citation header from a retrieved chunk's metadata.

        Uses ``page`` when present (set by the chunker for every chunk).
        Falls back to a ``page_start``/``page_end`` range, then to "?".
        """
        meta = doc.metadata or {}
        filename = meta.get("filename") or "unknown"
        section = meta.get("section") or "—"
        page = meta.get("page")
        ps = meta.get("page_start")
        pe = meta.get("page_end")
        if page is not None:
            page_str = f"p. {page}"
        elif ps is not None and pe is not None:
            page_str = f"p. {ps}" if ps == pe else f"pp. {ps}-{pe}"
        else:
            page_str = "p. ?"
        return f"[{i} document] {filename} — Section: {section} ({page_str})"

    def _format_doc_header_for_humans(self, doc: Document) -> str:
        meta = doc.metadata or {}
        filename = meta.get("filename") or "unknown"
        section = meta.get("section") or "—"
        page = meta.get("page")
        ps = meta.get("page_start")
        pe = meta.get("page_end")
        if page is not None:
            page_str = f"p. {page}"
        elif ps is not None and pe is not None:
            page_str = f"p. {ps}" if ps == pe else f"pp. {ps}-{pe}"
        else:
            page_str = "p. ?"
        return f"<span style='color: gray;'>[{filename} — Section: {section} ({page_str})]</span>"

    def generate(self,
                 messages: list[dict],
                 documents: list[Document],
                 max_new_tokens: int = 500,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 ) -> str:

        match self.model_name:
            # ------------------------------------------------------------
            # -----ministral/Ministral-3b-instruct------------------------
            # ------------------------------------------------------------
            case "ministral/Ministral-3b-instruct" | "HuggingFaceTB/SmolLM2-360M-Instruct" | "Qwen/Qwen2.5-0.5B-Instruct":
                # print(f"Conversation: {messages}")

                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.tokenizer(
                    prompt, return_tensors="pt").to(self.model.device)

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

            # ------------------------------------------------------------
            # -----google/gemma-3-4b-it-----------------------------------
            # ------------------------------------------------------------
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
                decoded = self.processor.decode(
                    generation, skip_special_tokens=True)
                print("DEBUG: Raw model output\n", decoded)
                return self.replace_model_references(decoded, documents)
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            case _:
                raise ValueError(f"Unsupported model name: {self.model_name}")

    def format_prompt(self, conversation: list[dict], documents: list[Document]):
        match self.model_name:


            # ------------------------------------------------------------
            # -----Models suitable for local testing
            # ------------------------------------------------------------
            case "ministral/Ministral-3b-instruct" | "HuggingFaceTB/SmolLM2-360M-Instruct" | "Qwen/Qwen2.5-0.5B-Instruct":
                conversation = [
                    {"role": "system", "content": MAIN_ASSISTANT_PROMPT}] + conversation

                conversation[-1]["content"] += "\n\nRetrieved documents:"

                for i, doc in enumerate(documents):
                    header = self._format_doc_header(i, doc)
                    conversation[-1]["content"] += f"\n{header}\n{doc.page_content}\n"

                if len(documents) == 0:
                    conversation[-1]["content"] += f"No relevant documents were found.\n"

                final_prompt = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                print("DEBUG: Prompt\n", final_prompt)
                return final_prompt

            # ------------------------------------------------------------
            # -----google/gemma-3-4b-it-----------------------------------
            # ------------------------------------------------------------
            case "google/gemma-3-4b-it":
                # import json
                # print(json.dumps(conversation, indent=2))
                for i, msg in enumerate(conversation):
                    conversation[i]["content"] = [
                        {"type": "text", "text": msg["content"]}]

                conversation = [{"role": "system", "content": [
                    {"type": "text", "text": MAIN_ASSISTANT_PROMPT}]}] + conversation

                conversation[-1]["content"][0]["text"] += "\n\nRetrieved documents:"

                for i, doc in enumerate(documents):
                    header = self._format_doc_header(i, doc)
                    conversation[-1]["content"][0]["text"] += f"\n{header}\n{doc.page_content}\n"

                if len(documents) == 0:
                    conversation[-1]["content"][0]["text"] += f"No relevant documents were found.\n"
                print("DEBUG: Conversation with document headers:\n", json.dumps(conversation, indent=2))
                return self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device, dtype=torch.float16)

            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            case _:
                raise ValueError(f"Unsupported model name: {self.model_name}")


    def replace_model_references(self, text: str, chunks:list[Document]) -> str:
        """
        Replace the model generated "<{n}>" references in the text with the corresponding document headers.

        """
        for i, doc in enumerate(chunks):
            header = self._format_doc_header_for_humans(doc)
            text = text.replace(f"<{{{i}}}>", header)
        return text



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
                

            case "sentence-transformers/all-MiniLM-L6-v2":
                self.model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu"
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
                
            case "sentence-transformers/all-MiniLM-L6-v2":
                return self.model.encode(input).tolist()

            case _:
                raise ValueError(f"Unsupported model name: {self.model_name}")


if __name__ == "__main__":
    qwen_local = "Qwen/Qwen2.5-0.5B-Instruct"
    huggy = "HuggingFaceTB/SmolLM2-360M-Instruct"
    llm = LLM(huggy)
    messages = [
        {"role": "user", "content": "Explain what attention is in a transformer model."}
    ]
    print("prompt: ")
    llm.format_prompt(messages, [])
    print("response: ")
    print(llm.generate(messages, []))

    # embedding_model = Embedder("sentence-transformers/all-MiniLM-L6-v2")

    # texts = [
    #     "Attention is cool",
    #     "Some other text"
    # ]

    # embeddings = embedding_model.encode(texts, is_query=True)
    # print(embeddings[0][:10])
    # embeddings = embedding_model.encode(texts, is_query=False)
    # print(embeddings[0][:10])
    while True:
        pass
