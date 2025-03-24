from vllm import SamplingParams, LLM

def main():
    llm = LLM('microsoft/Phi-4-mini-instruct', max_seq_len_to_capture=18496)
    prompt = 'An increasing sequence: one,'

    sampling_params = SamplingParams(max_tokens=50, temperature=0.0, top_p=0.0)
    outputs = llm.generate(prompt, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
if __name__ == '__main__':
    main()