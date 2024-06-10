from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class StableBeluga:
    def __init__(self):
        model_name = "stabilityai/StableBeluga2"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.system_prompt = "### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n### User: {message}\n\n### Assistant:\n"

        self.tokens_to_check = self.tokenizer.convert_tokens_to_ids(["▁No","▁Yes"])
        self.tokens_to_check_set = set(self.tokens_to_check)

    def run_generation(self, message, do_sample=False, max_new_tokens=256, temperature=1.0, top_p=1.0, return_dict_in_generate=True, output_scores=False):
        prompt = self.system_prompt.format(message=message)
        inputs = self.tokenizer(prompt, truncation=True, return_tensors="pt").to("cuda")

        output = self.model.generate(
            **inputs,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
        )
        output_ids = output.sequences
        
        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        
        outputs = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        
        return outputs
    
    def run_binary(self, message, do_sample=False, max_new_tokens=5, temperature=1.0, top_p=1.0, return_dict_in_generate=True, output_scores=True):
        prompt = self.system_prompt.format(message=message)
        inputs = self.tokenizer(prompt, truncation=True, return_tensors="pt").to("cuda")

        output = self.model.generate(
            **inputs,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
        )
        
        output_ids = output.sequences
        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs.input_ids[0]) :]
        outputs = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        # print(output.sequences)
        # print(output_ids)
        # print(outputs)
        
        # get normalized prediction
        scores = output.scores
        # retrieve the correct ids
        valid_ids = [i for i,pid in enumerate(output_ids) if pid.item() in self.tokens_to_check_set]

        pred, pred_norm = None, None

        if len(valid_ids) == 1:
            score = scores[valid_ids[0]]
            metric_scores = score[0,self.tokens_to_check]
            pred = metric_scores.argmax(dim=-1).item()

            # since binary we can just use softmax's result
            metric_scores = torch.nn.functional.softmax(metric_scores, dim=-1).detach().cpu()
            metric_scores = metric_scores[1]
            pred_norm = metric_scores.item()
        else:
            print("invalid ids", valid_ids, outputs)
            print("Check output.sequences and output_ids. Sometimes the yes no can be ▁No or ▁Yes")
        return pred, pred_norm