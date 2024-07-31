import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModel, CLIPImageProcessor, CLIPVisionModel

from examples.common import load_model
from easydict import EasyDict

args = {'model_version': 'radio_v2',
        'adaptor_name': 'openai_clip',
        'vitdet_window_size': None,
        'force_reload': False,
        'torchhub_repo': 'NVlabs/RADIO',
        'use_huggingface': False}

args = EasyDict(args)

radio_model, preprocessor, info = load_model(args.model_version, adaptor_names=args.adaptor_name, return_spatial_features=False,
                                           vitdet_window_size=args.vitdet_window_size, force_reload=args.force_reload,
                                           torchhub_repo=args.torchhub_repo, use_huggingface=args.use_huggingface)
radio_model.eval()

print(preprocessor)

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

model.vision_tower = radio_model.to(0)

processor = AutoProcessor.from_pretrained(model_id)

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