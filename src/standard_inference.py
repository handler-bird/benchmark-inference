from transformers import TextIteratorStreamer
from threading import Thread
from src.utils.load_model import LargeLanguageModel
import time
import pandas as pd


def standard_streaming(model: str, prompt: str, device: str, base_dir: str):
    llm = LargeLanguageModel(model_path=model, device=device, standard=True)

    streamer = TextIteratorStreamer(llm.tokenizer)

    inputs = llm.tokenizer([prompt], return_tensors="pt").to(device)

    generation_kwargs = {
        **inputs,
        "max_new_tokens": 50,
        "temperature": 0.0,
        "do_sample": False,
        "streamer": streamer,
    }

    thread = Thread(target=llm.model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    generated_list = []
    time_for_first_token_start = None
    time_total_generation_start = time.time()

    for new_text in streamer:
        if time_for_first_token_start is None:
            time_for_first_token_start = time_total_generation_start
            time_for_first_token_end = time.time() - time_for_first_token_start

        generated_text += new_text
        generated_list.append(new_text)

    time_total_generation_end = time.time() - time_total_generation_start

    print("generated text:", generated_text)
    print("time for first token:", time_for_first_token_end)
    print("total time for text:", time_total_generation_end)
    print("time per token", time_total_generation_end / len(generated_list))

    data = {
        "model": [model],
        "prompt": [prompt],
        "output": [generated_text],
        "time_to_first_token": [time_for_first_token_end],
        "time_per_token": [time_total_generation_end / len(generated_list)],
        "total_time": [time_total_generation_end],
    }

    df = pd.DataFrame(data)
    df.to_csv(f"{base_dir}/metrics.csv", index=False)
