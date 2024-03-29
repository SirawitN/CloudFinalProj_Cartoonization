import logging
import torch, os
import numpy as np
from PIL import Image
from model import Transformer
import torchvision.utils as vutils
import torchvision.transforms as transforms

MODEL_FILE_NAME = "hayao_model.pth"
LOAD_SIZE = 756

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    logger.info('Loading the model.')
	
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(model_dir, MODEL_FILE_NAME)
	
    model = Transformer()
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()
    logger.info('Done loading model.')
    return model

def predict_fn(input_data, model):
    logger.info('Generating cartoonization.')
	
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
    ext = os.path.splitext(input_data)[1]
    if ext not in ['.jpg', '.png']:
        logger.info('Invalid input type.')
        return None
    
    input_image = Image.open(input_data).convert("RGB")
	
    # resize image, keep aspect ratio
    h = input_image.size[0]
    w = input_image.size[1]
    ratio = h *1.0 / w
    if ratio > 1:
        h = LOAD_SIZE
        w = int(h*1.0/ratio)
    else:
        w = LOAD_SIZE
        h = int(w * ratio)
    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)

    # RGB -> BGR
    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
	
    # preprocess, (-1, 1)
    input_image = -1 + 2 * input_image 
    
    with torch.no_grad():
        input_data = input_data.to(device)
        cartoonized = model(input_data)

    output_image = cartoonized[0]
	# BGR -> RGB
    output_image = output_image[[2, 1, 0], :, :]
	# deprocess, (0, 1)
    output_image = output_image.data.cpu().float() * 0.5 + 0.5

    return output_image