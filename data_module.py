from torch.utils.data import Dataset
import torch
import pandas as pd
from template import get_llama3_chat_template



def convert_raw_data_to_model_qa(tokenizer, max_length, question, answer):
    """
    prepares input and labeled for the model based on the specified format
    """
    # if configs['model_family'] == 'llama3-8b-instruct':
    new_question = get_llama3_chat_template(instruction=question)
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

class QAForgetDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length = 512):
        """
        Initializes the dataset for gradient ascent finetuning

        Args:
            data_path (str): path to the data file. csv file containing columns 'question' and 'answer'
            tokenizer (transformers.PreTrainedTokenizer): tokenizer to process the input
            configs (dict): configuration settings, including model family (like Llama3) and max_length
            max_length (int, optional): maximum sequence length for tokenization. Defaults to 512.
        
        Returns:
            tuple of tensors
        """

        super(QAForgetDataset, self).__init__()
        self.data_path = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        #self.configs = configs
        self.max_length = max_length

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        question = self.data_path.iloc[idx]['question']
        answer = self.data_path.iloc[idx]['answer']
        return convert_raw_data_to_model_qa(tokenizer   = self.tokenizer, 
                                            max_length  = self.max_length, 
                                            question    = question, 
                                            answer      = answer, 
                                            )#configs     = self.configs
    
def custom_data_collator_forget(samples):
    """
    Collate function for the forget dataset only

    Args:
        samples (list of tuples): Each tuple contains (input_ids, labels, attention_mask)

    Returns:
        dict: batched_inputs, labels, attention_masks.

    """
    input_ids = torch.stack([sample[0] for sample in samples])
    labels = torch.stack([sample[1] for sample in samples])
    attention_mask = torch.stack([sample[2] for sample in samples])
    return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}

