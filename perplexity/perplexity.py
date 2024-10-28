import argparse
import datasets
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from longppl.longppl import *
import os


def compute_perplexity(
    encodings, model, evaluator_model, tokenizer, evaluator_tokenizer, args, device=None
):
    if device is not None:
        assert device in ["gpu", "cpu",
                          "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoded_texts = [x[0:args.max_length-1] for x in encodings["input_ids"]]

    pbar = tqdm(total=len(encoded_texts))
    longppls, ppls, nums_key_token, nums_token = [], [], [], []

    def convert_tokenized_to_text(tokenized_input, tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        text = tokenizer.batch_decode(tokenized_input)
        return text

    for encoding_index in range(0, len(encoded_texts)):
        tokenized_input = torch.tensor(encoded_texts[encoding_index:encoding_index+1]).to(device)
        if args.tokenized:
            text = convert_tokenized_to_text(tokenized_input, args.llama_path)
        else:
            text = convert_tokenized_to_text(tokenized_input, args.model)
        save_path = os.path.join("key_text", args.evaluator_name, f"slice_{encoding_index}.txt")

        with torch.no_grad():
            output = compute_longppl(
                text=text[0], 
                model=model,
                evaluator_model=evaluator_model,
                tokenizer=tokenizer, 
                evaluator_tokenizer=evaluator_tokenizer, 
                save_path=save_path, 
                trunc_len=args.trunc_len, 
                sliding_window=args.sliding_window
            )
        longppl = output['longppl']
        ppl = output['ppl']
        n_key_token = output['n_key_token'] 
        n_token = output['n_token']
        
        if longppl is not None:
            longppls.append(longppl)
            nums_key_token.append(n_key_token)
        ppls.append(ppl)
        nums_token.append(n_token)
        longppl = (np.stack(longppls) * np.stack(nums_key_token)).sum() / np.stack(nums_key_token).sum()
        ppl = (np.stack(ppls) * np.stack(nums_token)).sum() / np.stack(nums_token).sum()

        pbar.set_postfix(longppl=longppl, ppl=ppl)
        pbar.update(1)

    return {"longppl": longppl, "ppl": ppl}



def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if args.mode == 'online':
        evaluator_model = AutoModelForCausalLM.from_pretrained(args.evaluator_model, torch_dtype=torch.bfloat16, device_map="auto")
    elif args.mode == 'offline':
        evaluator_model = None
    evaluator_tokenizer = AutoTokenizer.from_pretrained(args.evaluator_model)

    if args.tokenized:
        try:
            input_texts = datasets.load_from_disk(args.tokenized)
        except:
            input_texts = datasets.load_dataset(
                args.tokenized, name=args.subset, split=args.split)
    else:
        input_texts = datasets.load_dataset(
            args.dataset, name=args.subset, split=args.split)

        def tokenize(example):
            tokenized = tokenizer(
                example[args.feature],
                add_special_tokens=False,
                padding=True,
                truncation=False,
                max_length=sys.maxsize,
                return_attention_mask=True,
                return_offsets_mapping=True
            )
            example["input_ids"] = tokenized["input_ids"]
            example["attention_mask"] = tokenized["attention_mask"]
            example["tokenized_len"] = len(tokenized["input_ids"])
            example["offsets_mapping"] = tokenized["offsets_mapping"]
            return example

        input_texts = input_texts.map(tokenize)
        if args.save_tokenized:
            input_texts.save_to_disk(args.save_tokenized)
            print(f"Saved tokenized dataset to {args.save_tokenized}")
            return

    if args.dataset_min_tokens:
        input_texts = input_texts.filter(
            lambda x: x["tokenized_len"] >= args.dataset_min_tokens)
    if args.samples:
        input_texts = input_texts['test'][:args.samples]
    

    ppl = compute_perplexity(
        model=model, 
        evaluator_model=evaluator_model, 
        tokenizer=tokenizer, 
        evaluator_tokenizer=evaluator_tokenizer,
        encodings=input_texts,
        args=args,
    )
    print(f"{args.model}: longppl: {ppl['longppl']}, ppl: {ppl['ppl']}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--evaluator-model", type=str)
    parser.add_argument("--evaluator-name", type=str, help='To use the offline key tokens we provided, set it to Qwen2-72B-Instruct, Mistral-Large-Instruct-2407, or Meta-Llama-3.1-8B', default="Meta-Llama-3.1-8B")
    parser.add_argument("--mode", type=str, choices=['online', 'offline'], default='offline')
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-s", "--subset", type=str)
    parser.add_argument("-f", "--feature", type=str)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--dataset-min-tokens", type=int)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--save-tokenized", type=str)
    parser.add_argument("--tokenized", type=str)
    parser.add_argument("--trunc-len", type=int, default=4096)
    parser.add_argument("--sliding-window", type=int, default=1024)
    parser.add_argument("--llama-path", type=str, default="Llama-2-7b-hf")
    main(parser.parse_args())