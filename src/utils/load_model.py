from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from vllm import LLM


class LargeLanguageModel:
    def __init__(
        self,
        device: str,
        model_path: str = "microsoft/Phi-4-mini-instruct",
        strategy: str = 'standard'
    ):
        torch.random.manual_seed(0)

        if strategy == 'standard':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if strategy == 'bfloat16':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if strategy == 'quantization':
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if strategy == 'flash_attn':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if strategy == 'vllm':
            self.model = LLM(model=model_path, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if strategy == 'unsloth':
            from unsloth import FastLanguageModel

            paths = model_path.split("/")
            model_path = "unsloth/" + paths[-1]
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path, load_in_4bit=False, max_seq_length=8192, dtype=torch.float16
            )

            FastLanguageModel.for_inference(self.model)
