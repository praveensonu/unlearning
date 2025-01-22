from transformers import Trainer
import torch



class ForgetTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the gradient ascent loss for the model
        """
        #if self.loss_type == 'grad_ascent':
        # unpack the forget inputs
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        attention_mask = inputs['attention_mask']

        # forward pass
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        forget_loss = outputs.loss * -1 # gradient ascent is negating the loss

        loss = forget_loss
        return (loss, outputs) if return_outputs else loss



class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss') 
        self.cfg = kwargs.pop('cfg')
        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the gradient ascent loss for the model
        """
        if self.loss_type == 'grad_ascent':
            # unpack the forget inputs
            input_ids = inputs['input_ids']
            labels = inputs['labels']
            attention_mask = inputs['attention_mask']

            # forward pass
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels
            )
            forget_loss = outputs.loss * -1 # gradient ascent is negating the loss

            loss = forget_loss
            


        elif self.loss_type == 'grad_diff':
            # unpack the forget inputs
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels
            )
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(
                input_ids = retain_input_ids,
                attention_mask = retain_attention_mask,
                labels = retain_labels
            )
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss
            
        return (loss, outputs) if return_outputs else loss

            
    def prediction_step(self, model, inputs, prediction_loss_only:bool, ignore_keys=None):
        """
        predictions during evaluation
        """
        forget_inputs = inputs 
        inputs_ids, labels, attention_mask = forget_inputs

        # forward pass
        with torch.no_grad():
            outputs = model(
                input_ids = inputs_ids,
                attention_mask = attention_mask,
                labels = labels
            )
            logits = outputs.logits
            loss = outputs.loss
            return (loss, logits, labels)