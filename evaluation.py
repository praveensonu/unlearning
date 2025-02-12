import os

# please change this with the availabel gpus
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval_utils import *

forget_10 = load_dataset('locuslab/TOFU', "forget10")
forget_10_perturb = load_dataset('locuslab/TOFU', "forget10_perturbed")


print("\n\nNow running forget_10 model......")

# Please change this model for different unlearn methods
revision = "checkpoint-60"
forget_model_id = 'locuslab/phi_grad_diff_1e-05_forget10'
forget_tokenizer = AutoTokenizer.from_pretrained(forget_model_id, revision= revision)
forget_model = AutoModelForCausalLM.from_pretrained(forget_model_id,  revision= revision) #torch_dtype = torch.bfloat16,


device = "cuda" if torch.cuda.is_available() else "cpu"
forget_model.to(device)
forget_model.eval()

question_start_token = "Question: "
question_end_token = "\n"
answer_token = "Answer: "


# on the paraphrased answer
forget_para_dataloader = create_dataloader(forget_10_perturb, forget_tokenizer, forget_model, 'paraphrased_answer')
# on the perturbed answers
forget_perturbed_dataloader = create_dataloader(forget_10_perturb, forget_tokenizer, forget_model, 'perturbed_answer')


print("\n\nGetting losses from the forget10 model......")
forget_logs = get_eval_logs(para_data=forget_para_dataloader, 
                          perturbed_data=forget_perturbed_dataloader, 
                          model=forget_model)

# unlearned_evals = "path/to/save"
# with open(unlearned_evals, 'w') as f:
#     json.dump(forget_logs, f, indent = 4)

del forget_model
torch.cuda.empty_cache()
torch.cuda.synchronize()


print("\nFinished up forget10......")
print("\n\nNow running retain10 model......")
## retain90 starts here
retain_model_id = 'locuslab/tofu_ft_retain90_phi-1.5'

# I tried using the retain90 model but it was giving configuration_phi.py error, the phi model repo is updated and removed this file. 
# even removing trust_remote_code, or updating the transformers lib, or downgrading it to 4.44.2 couldnt solve it.
retain_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code = True) 
retain_model = AutoModelForCausalLM.from_pretrained(retain_model_id)

retain_model.to(device)
retain_model.eval()

retain_para_dataloader = create_dataloader(forget_10_perturb, retain_tokenizer, retain_model, 'paraphrased_answer')
# on the perturbed answers
retain_perturbed_dataloader = create_dataloader(forget_10_perturb, retain_tokenizer, retain_model, 'perturbed_answer')

print("\n\nGetting losses from the retain90 model......")
retain_logs = get_eval_logs(para_data=forget_para_dataloader, 
                          perturbed_data=forget_perturbed_dataloader, 
                          model=retain_model)


print("\n\nCalculating thhe forget quality")
forget_quality = cal_forget_quality(forget_logs, retain_logs)
print(forget_quality)