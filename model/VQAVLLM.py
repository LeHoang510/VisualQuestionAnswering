from transformers import AutoProcessor
from transformers import LlavaForConditionalGeneration
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
import torch
from PIL import Image

class VQAVLLM():
    def __init__(self, device):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", 
                                                                   quantization_config=quantization_config, 
                                                                   device_map=device)
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.generation_config = GenerationConfig(max_new_tokens=10,
                                                  do_sample=True,
                                                  temperature=0.1,
                                                  top_p=0.95,
                                                  top_k=50,
                                                  eos_token_id=self.model.config.eos_token_id,
                                                  pad_token_id=self.model.config.pad_token_id,)
        
    def predict(self, img_path, text):
        img = Image.open(img_path).convert("RGB")
        text = self.create_prompt(text)
        inputs = self.processor(images=img,
                                text=text,
                                return_tensors="pt",
                                padding=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            assistant_response = generated_text.split("### ASSISTANT:")[-1].strip() 
            
        return assistant_response

        
    def create_prompt(self, question):
        prompt = """### INSTRUCTION:
        Your task is to answer the question based on the given image. 
        You can only answer 'yes' or 'no'.
        ### USER: <image>
        {question}
        ### ASSISTANT:"""
        return prompt
