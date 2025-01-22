
class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.LoRA_r         = 8
        self.LoRA_alpha     =32
        self.LoRA_dropout   =0.05
        self.lr             = 1e-5
        self.batch_size= 32
        self.gradient_accumulation_steps = 1
        self.num_epochs = 100
        self.forget_loss = 'grad_ascent' #select from grad_ascent, grad_diff, KL, dpo
        self.overwrite_dir = True
        self.weight_decay = 0.01 
        self.save_dir = '/home/praveen/theoden/ul_paper/outputs/testing'
        self.access_token = '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'
        self.model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.data_path = '/home/praveen/theoden/ul_paper/dataset/forget.csv'


