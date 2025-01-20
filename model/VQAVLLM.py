from transformers import AutoProcessor
from transformers import LlavaForConditionalGeneration
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
import torch
from PIL import Image

class VQAVLLM():
    def __init__(self, device):
        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf").to(device)
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.generation_config = GenerationConfig(max_new_tokens=10,
                                                  do_sample=True,
                                                  temperature=0.1,
                                                  top_p=0.95,
                                                  top_k=50,
                                                  eos_token_id=self.model.config.eos_token_id,
                                                  pad_token_id=self.model.config.pad_token_id,)

    def predict(self, img, text):
        inputs = self.processor(images=img,
                                text=text,
                                return_tensors="pt",
                                padding=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)

        return generated_text