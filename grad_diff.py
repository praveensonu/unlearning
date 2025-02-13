import torch
from template import LLAMA3_CHAT_TEMPLATE
from torch.utils.data import Dataset
from transformers import Trainer
from data_module import convert_raw_data_to_model_qa




def convert_raw_data_to_model_qa(tokenizer, max_length, question, answer):
    """
    prepares input and labeled for the model based on the specified format
    """
    # if configs['model_family'] == 'llama3-8b-instruct':
    new_question = LLAMA3_CHAT_TEMPLATE.format(instruction=question)
    # else:
    #     raise ValueError(f"Invalid model_family: {configs['model_family']}")
    
    full_text = new_question + answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )

    pad_length = max_length - len(encoded['input_ids'])
    pad_input_ids = encoded['input_ids'] + [tokenizer.pad_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

    if len(encoded['input_ids']) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)

    # Mask out the question tokens in the labels
    for i in range(num_question_tokens):
        label[i] = -100

    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)




class DualDataset(Dataset):

    """
    Data set class for creating data for forget and retain (which is used by gradient difference)

    Args:
        forget: forget dataset
        retain: retain dataset
        tokenizer: tokenizer
        max_length: max length

    Returns something like this:
        (
        ([input_ids], [labels], [attention_mask]), # forget date for sample 1
        ([input_ids], [labels], [attention_mask]),# retain data for sample 1
        ([input_ids], [labels], [attention_mask]), # forget data for sample 2
        ([input_ids], [labels], [attention_mask]) # retain data for sample 2
        ) 

    """
    def __init__(self, forget, retain, tokenizer, max_length):
        self.forget = forget.reset_index(drop=True)
        self.retain = retain.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return max(len(self.forget), len(self.retain))
    
    def __getitem__(self, idx):
    #doing this to have a cyclic rotation of the data. if idx =2 and len of retain is 10, then 2 % 10 =2

        forget_idx = idx % len(self.forget)
        retain_idx = idx % len(self.retain)

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.forget.iloc[forget_idx]['question'],
            self.forget.iloc[forget_idx]['answer']
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.retain.iloc[retain_idx]['question'],
            self.retain.iloc[retain_idx]['answer']
        )

        return (
            (forget_data[0], forget_data[1], forget_data[2]),
            (retain_data[0], retain_data[1], retain_data[2])
        )
    


def custom_gd_collator_forget(samples):
    """
    Custom data collator for forget and retain data

    Args:
        samples: list of tuples (forget_data, retain_data) from the DualDataset class

    Returns:
        rets: list of tuples (input_ids, labels, attention_mask)
        example output for batch size 2
        
        [(  #forget data for batch of 2
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # input_ids
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # labels
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # attention_mask
            ),
            (  #retain data for batch of 2
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # input_ids
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # labels
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # attention_mask
            ),
        ]

    """

    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets


class GradDiffTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs = False):
        forget_inputs, retain_inputs = inputs
        input_ids, labels, attention_mask = forget_inputs

        ## gradient ascent on the forget
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        forget_loss = outputs.loss
        forget_loss = forget_loss * -1

        ## gradient descent on the retain
        retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
        retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
        retain_loss = retain_outputs.loss
        loss = forget_loss + retain_loss

        return (loss, outputs) if return_outputs else loss