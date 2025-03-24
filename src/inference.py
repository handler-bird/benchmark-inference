#TODO: Remove when fixed
#import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["VLLM_USE_SPAWN"] = "1"
#os.environ["VLLM_WORKERS"] = "1"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

#TODO: Remove when fixed
#import multiprocessing
#if multiprocessing.get_start_method() != 'spawn':
#    multiprocessing.set_start_method('spawn', force=True)

import torch
from threading import Thread
from src.utils.load_model import LargeLanguageModel
from src.utils.functions import save_metrics
import time
from src.utils.gpu import GPU

def inference_streaming(strategy: str, model: str, prompt: str, device: str, base_dir: str):
    #TODO: Remove when fixed
    #print(f"In inference_streaming, multiprocessing start method: {multiprocessing.get_start_method()}")
    
    if strategy == "unsloth":
        import unsloth
        from transformers import TextIteratorStreamer
    else:
        from transformers import TextIteratorStreamer
    
    # Start GPU monitoring
    gpu = GPU()
    gpu.start_measure()

    #INFO: This fixes the erro with vllm RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    if device == "cuda":
        torch.cuda.init()
        print(f"CUDA initialized, available: {torch.cuda.is_available()}")

    llm = LargeLanguageModel(model_path=model, device=device, strategy=strategy)

    streamer = TextIteratorStreamer(llm.tokenizer)

    inputs = llm.tokenizer([prompt], return_tensors="pt").to(device)

    generation_kwargs = {
        **inputs,
        "max_new_tokens": 50,
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
    time_per_token = time_total_generation_end / len(generated_list)

    # Stop GPU monitoring and join inference thread to free up resources and ensure it has finished
    thread.join()
    gpu.stop_measure()

    gpu_peak_memory = gpu.get_memory_usage(peak=True)
    gpu_min_memory = gpu.get_memory_usage()
    gpu_peak_utilization = gpu.get_utilization(peak=True)
    gpu_min_utilization = gpu.get_utilization()

    print("generated text:", generated_text)
    print("time for first token:", time_for_first_token_end)
    print("total time for text:", time_total_generation_end)
    print("time per token", time_total_generation_end / len(generated_list))
    print(f"GPU memory usage (min, max) in MB: ({gpu_min_memory}, {gpu_peak_memory})")
    print(f"GPU utilization percentage (min, max): ({gpu_min_utilization}, {gpu_peak_utilization})")

    save_dir = base_dir + '/metrics.csv'
    save_metrics(
        save_dir=save_dir,
        model=model,
        prompt=prompt,
        output=generated_text,
        time_to_first_token=time_for_first_token_end,
        time_per_token=time_per_token,
        total_time=time_total_generation_end,
        gpu_min_memory=gpu_min_memory,
        gpu_peak_memory=gpu_peak_memory,
        gpu_min_utilization=gpu_min_utilization,
        gpu_peak_utilization=gpu_peak_utilization
    )
