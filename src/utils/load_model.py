import torch

class LargeLanguageModel:
    def __init__(
        self,
        device: str,
        model_path: str = "microsoft/Phi-4-mini-instruct",
        strategy: str = 'standard'
    ):
        torch.random.manual_seed(0)

        if strategy == 'standard':
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if strategy == 'bfloat16':
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if strategy == 'quantization':
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if strategy == 'flash_attn':
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if strategy == 'vllm':
            import vllm
            from transformers import AutoTokenizer

            #TODO: https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html#python-multiprocessing
            self.model = vllm.LLM(model=model_path, trust_remote_code=True, quantization="fp8")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if strategy == 'unsloth':
            import unsloth
            from unsloth import FastLanguageModel

            paths = model_path.split("/")
            model_path = "unsloth/" + paths[-1]
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path, load_in_4bit=False, max_seq_length=8192, dtype=torch.float16
            )

            FastLanguageModel.for_inference(self.model)
