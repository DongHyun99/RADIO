from easydict import EasyDict
from radio_vision_tower import RADIOVisionTower
from transformers import AutoTokenizer, BitsAndBytesConfig, LlavaProcessor, LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

import torch
import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "llava-hf/llava-1.5-7b-hf" # "liuhaotian/llava-v1.5-7b"
args = {'mm_im_crop': True} # for  CLIPImageProcessor
args = EasyDict(args)

model = LlavaForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor2 = AutoProcessor.from_pretrained(model_path)

# # radio:<image_size>:<checkpoint_or_version>:<extra_config>
vision_tower = RADIOVisionTower('radio:432:radio_v1', args)
vision_tower.to(device=device, dtype=torch.float16)

image_processor = vision_tower.image_processor
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
processor = LlavaProcessor(image_processor, tokenizer, chat_template=processor2.chat_template)

conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What are these?"},
          {"type": "image"},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))