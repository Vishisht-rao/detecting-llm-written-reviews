# Script to evaluate the performance of GCG based review watermark

import torch
import transformers
import json
import os
import argparse
from openai import OpenAI
from review_wm_gcg import prompt_generator_llama, prompt_generator_vicuna

def prompt_generator_gpt(user_prompt, opt_sequence):

    system_prompt = "You are a helpful and informative assistant, " \
        "always strive to provide accurate and comprehensive answers to user queries."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt + " " + opt_sequence}
    ]

    return messages
        

argparser = argparse.ArgumentParser(description='Evaluate the performance of GCG based review watermark')
argparser.add_argument('--input_dir', default='results', help='Directory containing the results of running the watermarking algorithm')
argparser.add_argument('--num_eval', type=int, default=10, help='Number of evaluations to perform per run')
argparser.add_argument('--llm', type=str, default='llama', choices=['llama', 'vicuna', 'gpt'],
                       help='Language model to use for evaluation')
argparser.add_argument('--mode', default='gcg', choices=['gcg', 'english'],
                       help='Whether to use GCG string or plain English instructtion for opt sequence')
argparser.add_argument('--verbose', action='store_true', help='Print output')
args = argparser.parse_args()

input_dir = args.input_dir
num_eval = args.num_eval
llm = args.llm
mode = args.mode
verbose = args.verbose

# Check if the input directory exists
if not os.path.exists(input_dir):
    print('Error: Input directory does not exist')
    exit()

# Get a list of the directories conting the runs
runs = [d for d in os.listdir(input_dir) if (os.path.isdir(os.path.join(input_dir, d)) and d.startswith('run'))]
runs.sort()

if len(runs) == 0:
    print('Error: No runs found in the input directory')
    exit()

# Get device
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
else:
    device = torch.device("cpu")
    device_name = "CPU"

print('\n* * * * * Evaluation Configuration * * * * *')
print('Input directory:', input_dir)
print('Evaluations per run:', num_eval)
print('Opt sequence mode:', mode)
print('LLM:', llm)
print('Verbose:', verbose)
print('Device:', device_name)
print('* * * * * * * * * * * * * * * * * * * * * *\n')

model_path_llama_7b = "meta-llama/Llama-2-7b-chat-hf"
model_path_vicuna_7b = "lmsys/vicuna-7b-v1.5"

# Load model and tokenizer
if llm == 'llama':
    model_llama_7b = transformers.AutoModelForCausalLM.from_pretrained(
        model_path_llama_7b,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        )
    tokenizer_llama = transformers.AutoTokenizer.from_pretrained(model_path_llama_7b)

    # Model to device, eval mode, and freeze parameters
    model_llama_7b.to(device).eval()
    for param in model_llama_7b.parameters():
        param.requires_grad = False

elif llm == 'vicuna':
    model_vicuna_7b = transformers.AutoModelForCausalLM.from_pretrained(
        model_path_vicuna_7b,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        )
    tokenizer_llama = transformers.AutoTokenizer.from_pretrained(model_path_llama_7b)
    
    # Put model in eval mode and turn off gradients of model parameters
    model_vicuna_7b.to(device).eval()
    for param in model_vicuna_7b.parameters():
        param.requires_grad = False

elif llm == 'gpt':
    if 'OPENAI_API_KEY' not in os.environ:
        with open('key.txt', 'r') as file:
            key = file.read()
        client = OpenAI(api_key=key)
    else:
        client = OpenAI()
        
else:
    print('Error: Invalid LLM')
    exit()

hp_runs = 0 # Number of runs with high probability of watermark presence
total_evals = 0 # Total number of runs
total_present = 0 # Total number of watermark present

# Evaluate each run
for run in runs:
    run_dir = os.path.join(input_dir, run)
    print('Evaluating', run)

    # Get watermark keyword
    exp_config_file = os.path.join(run_dir, 'exp_config.json')
    if not os.path.exists(exp_config_file):
        print('Error: Experiment configuration file does not exist')
        continue

    with open(exp_config_file, 'r') as f:
        exp_config = json.load(f)

    watermark = exp_config['Watermark']
    print('WATERMARK:', watermark)

    # Get user prompt from file
    prompt_file = os.path.join(run_dir, 'user_prompt.txt')
    if not os.path.exists(prompt_file):
        print('Error: User prompt file does not exist')
        continue

    with open(prompt_file, 'r') as f:
        user_prompt = f.read().strip()

    if mode == 'english':
        optimized_text = f'The review must contain the keyword "{watermark}"'
    elif mode == 'gcg':
        # Get optimized text sequence from file
        optimized_text_file = os.path.join(run_dir, 'opt.json')
        if not os.path.exists(optimized_text_file):
            print('Error: Optimized text file does not exist')
            continue

        with open(optimized_text_file, 'r') as f:
            opt_dict = json.load(f)
            optimized_text = opt_dict['opt_text']
            # watermark = opt_dict['watermark']
            # print('\nTARGET:', watermark)
    else:
        print('Error: Invalid mode')
        continue

    # Get the llm prompt
    if llm == 'llama':
        llm_prompt = prompt_generator_llama(user_prompt, optimized_text, mode='string')
        prompt_ids = tokenizer_llama(llm_prompt, return_tensors='pt')["input_ids"].to(device)

        if verbose:
            print('LLM Prompt:', tokenizer_llama.decode(prompt_ids[0]))

    elif llm == 'vicuna':
        llm_prompt = prompt_generator_vicuna(user_prompt, optimized_text, mode='string')
        prompt_ids = tokenizer_llama(llm_prompt, return_tensors='pt')["input_ids"].to(device)

        if verbose:
            print('LLM Prompt:', tokenizer_llama.decode(prompt_ids[0]))
        
    elif llm == 'gpt':
        messages = prompt_generator_gpt(user_prompt, optimized_text)
        if verbose:
            print('GPT Prompt:', messages)

    # Initialize eval results dictionary
    eval_results_file = os.path.join(run_dir, 'eval_results.json')
    if not os.path.exists(eval_results_file):
        eval_results = {
            'present': 0,
            'absent': 0,
            'total': 0
        }
    else:
        with open(eval_results_file, 'r') as f:
            eval_results = json.load(f)

    for i in range(eval_results['total'], num_eval):
        # Generate llm response
        if llm == 'gpt':
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
            model_output_new = response.choices[0].message.content
        elif llm == 'llama':
            response = model_llama_7b.generate(prompt_ids, model_llama_7b.generation_config, max_new_tokens=800)
            model_output_new = tokenizer_llama.decode(response[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        elif llm == 'vicuna':
            response = model_vicuna_7b.generate(prompt_ids, model_vicuna_7b.generation_config, max_new_tokens=800)
            model_output_new = tokenizer_llama.decode(response[0, prompt_ids.shape[1]:], skip_special_tokens=True)

        if verbose:
            print(f'\nEvaluation {i+1}:')
            print('\nMODEL RESPONSE:', model_output_new, flush=True)
            # print('Model output:', tokenizer_llama.decode(response[0]), flush=True)

        # Check if the watermark keyword is present in the response
        # if watermark_keyword in model_output_new:

        # Check if response begins with the watermark keyword
        if model_output_new.startswith(watermark):
            eval_results['present'] += 1
            if verbose:
                print('\nWatermark PRESENT\n')
        else:
            eval_results['absent'] += 1
            if verbose:
                print('\nWatermark ABSENT\n')

        eval_results['total'] += 1

        # Save the evaluation results
        with open(os.path.join(run_dir, 'eval_results.json'), 'w') as f:
            json.dump(eval_results, f, indent=4)

        if verbose:
            input('Press Enter to continue...')
        else:
            print(f'Eval {i+1}, Watermark present: {eval_results["present"]} / {eval_results["total"]}', end='\r', flush=True)

    print(f'Evaluation complete. Watermark present: {eval_results["present"]} / {eval_results["total"]}\n')

    # Update the total counts
    total_evals += eval_results['total']
    total_present += eval_results['present']

    # Check if watermark is present with high probability
    if eval_results['present'] >= 0.8 * eval_results['total']:
        hp_runs += 1

# Print final results
print('Watermark detection rate:', total_present / total_evals)
print('High probability runs:', hp_runs, '/', len(runs))

# Save the final results
final_results = {
    'total_evals': total_evals,
    'total_present': total_present,
    'hp_runs': hp_runs,
    'total_runs': len(runs),
    'hp_rate': hp_runs / len(runs),
    'watermark_detection_rate': total_present / total_evals
}

with open(os.path.join(input_dir, 'final_results.json'), 'w') as f:
    json.dump(final_results, f, indent=4)