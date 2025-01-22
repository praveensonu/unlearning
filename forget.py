import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import transformers
from data_module import *
from forget_trainer import CustomTrainerForgetting
import hydra


@hydra.main(version_base = None, config_path = 'config', config_name = 'config')
def main(cfg):
    pretrained_model = cfg.pretrained_model_path

    #Load the finetuned model
    ft_model = cfg.pretrained_model_path 
    model = AutoModelForCausalLM.from_pretrained(
        ft_model,
        torch_dtype = torch.bfloat16,
        trust_remote_code = True,
        token = cfg.access_token,
        device_map = 'auto',
    )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # configuring LoRA 
    config = LoraConfig(
        r = cfg.forget.LoRA_r,
        lora_alpha = cfg.forget.LoRA_alpha,
        lora_dropout= cfg.forget.LoRA_dropout,
        target_modules = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', ' q_proj', 'down_proj'],
        bias = 'none',
        task_type = 'CAUSAL_LM',
    )

    # wrapping the model with the LoRA configuration
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize the dataset
    data_path = cfg.data_path 
    dataset = QAForgetDataset(data_path = data_path, 
                              tokenizer = tokenizer,
                              configs = {'model_family': cfg.model_family},
                              max_length = cfg.max_length)
    
    # Initialize the dataloader with custom data collator
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size = cfg.forget.batch_size,
    #     shuffle = True,
    #     collate_fn = custom_data_collator_forget

    # )

    # Train the model
    training_args = transformers.TrainingArguments(
        output_dir = cfg.forget.save_dir,
        learning_rate = cfg.forget.lr,
        per_device_train_batch_size= cfg.forget.batch_size,
        per_device_eval_batch_size=  cfg.forget.batch_size,
        num_train_epochs= cfg.forget.num_epochs,
        weight_decay = cfg.forget.weight_decay,
        logging_dir = f'{cfg.forget.save_dir}/logs',
        #save_steps = cfg.forget.save_steps,
        evaluation_strategy= 'no',
        save_total_limit= 2,
        bf16 = True,

    )

    # Initialize the custom trainer
    trainer = CustomTrainerForgetting(
                model = model, 
                args = training_args,
                train_dataset = dataset,
                tokenizer = tokenizer,
                data_collator = custom_data_collator_forget,
                forget_loss = cfg.forget.forget_loss
    )

    # train the model
    model.config.use_cache = False
    trainer.train()

    # save the LoRA adapter
    model.save_pretrained(cfg.forget.save_dir)
    print(f'Forget LoRA adapter saved at {cfg.forget.save_dir}')

if __name__ == '__main__':
    main()