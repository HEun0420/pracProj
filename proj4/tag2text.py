# STEP 1 : import modules
import argparse
import numpy as np
import random
import torch
from PIL import Image
from ram.models import tag2text
from ram import inference_tag2text as inference
from ram import get_transform

# STEP 2: create inference object
model_path='Tag2Text inferece for tagging and captioning'
model = tag2text(pretrained=model_path, image_size=384,vit='swin_b')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# STEP 3: Load data
image_path= 'tag2text_framework.png'
transform = get_transform(image_size=384)
image = transform(Image.open(image_path)).unsqueeze(0).to(device)


delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]

model = model.to(device)
    
# STEP 4: inference
res = inference(image, model)
print("Model Identified Tags: ", res[0])
print("User Specified Tags: ", res[1])
print("Image Caption: ", res[2])