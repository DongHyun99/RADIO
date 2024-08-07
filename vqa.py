import requests
from PIL import Image

import torch
from transformers import LlavaForConditionalGeneration
from radio_vision_tower import RADIOVisionTower
from examples.common import load_model
from easydict import EasyDict

args = {'mm_im_crop': False}
args = EasyDict(args)

# radio:<image_size>:<checkpoint_or_version>:<extra_config>
radio_vision_tower = RADIOVisionTower('radio:432:radio_v2', args)

model_id = "liuhaotian/llava-v1.5-7b"
# model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

model.vision_tower = radio_vision_tower

processor = radio_vision_tower.make_preprocessor_external()

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
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