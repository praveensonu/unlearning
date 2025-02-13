import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4"


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from grad_diff import DualDataset, custom_gd_collator_forget, GradDiffTrainer
from config import Config
from peft import PeftModel, LoraConfig, get_peft_model
from forget_trainer import ForgetTrainer
from perplexity import Perplexity
from data_module import  QAForgetDataset, custom_data_collator_forget

## using the llama 3 template here, we can later change it to Olmo's template for our experiments

LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


forget = pd.read_csv('/home/praveen/theoden/ul_paper/dataset/forget.csv')
retain = pd.read_csv('/home/praveen/theoden/ul_paper/dataset/retain.csv')


cfg = Config()

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(cfg.model_id, 
                                             device_map = 'auto',
                                             torch_dtype = torch.bfloat16, 
                                             token=cfg.access_token,)


config = LoraConfig(
        r = cfg.LoRA_r,
        lora_alpha = cfg.LoRA_alpha,
        lora_dropout= cfg.LoRA_dropout,
        target_modules = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', ' q_proj', 'down_proj'],
        bias = 'none',
        task_type = 'CAUSAL_LM',
    )
# wrapping the model with the LoRA configuration
model = get_peft_model(model, config)
model.print_trainable_parameters()



if cfg.loss_type == 'grad_diff':
    dataset = DualDataset(
        forget, retain, tokenizer, 266
    )
    training_args = TrainingArguments(
        output_dir = cfg.save_dir,
        learning_rate = cfg.lr,
        per_device_train_batch_size= 4, # for grad diff I used smaller batch size
        num_train_epochs= 10,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{cfg.save_dir}/logs',
        #save_steps = cfg.forget.save_steps,
        evaluation_strategy= 'no',
        save_total_limit= 2,
        bf16 = True,

    )

    trainer = GradDiffTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        tokenizer = tokenizer,
        data_collator = custom_gd_collator_forget,
    )

if cfg.loss_type == 'grad_ascent' :
    dataset = QAForgetDataset(data_path = cfg.data_path,
                          tokenizer = tokenizer,
                          max_length = 266) 
    

    training_args = TrainingArguments(
        output_dir = cfg.save_dir,
        learning_rate = cfg.lr,
        per_device_train_batch_size= cfg.batch_size,
        per_device_eval_batch_size=  cfg.batch_size,
        num_train_epochs= 10,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{cfg.save_dir}/logs',
        #save_steps = cfg.forget.save_steps,
        evaluation_strategy= 'no',
        save_total_limit= 2,
        bf16 = True,)
    
    trainer = ForgetTrainer(
            model = model, 
            args = training_args,
            train_dataset = dataset,
            tokenizer = tokenizer,
            data_collator = custom_data_collator_forget,)


model.config.use_cache = False
trainer.train()

model.merge_and_unload()
model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
print(f'Forget LoRA adapter saved at {cfg.save_dir}')

batch_size = cfg.batch_size
max_length = 266

## perplexity on forget set after unlearning
## -> conditional perplexity calculation on answer given a question
qa_perplexity_ul = Perplexity(
    model = model, 
    tokenizer =tokenizer, 
    template =LLAMA3_CHAT_TEMPLATE, 
    batch_size = batch_size, 
    max_length =max_length,
    df =forget,
    case='qa',
    chat_tokens=4)

print(qa_perplexity_ul)


## perplexity on retain after unlearning
## -> conditional perplexity calculation on answer given a question

qa_perplexity_ul = Perplexity(
    model = model, 
    tokenizer =tokenizer, 
    template =LLAMA3_CHAT_TEMPLATE, 
    batch_size =batch_size, 
    max_length =max_length,
    df = retain,
    case='qa',
    chat_tokens=4)

print(qa_perplexity_ul)

