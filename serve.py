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



for files in os.listdir(input_dir):
	ext = os.path.splitext(files)[1]
	if ext not in ['.jpg', '.png']:
		continue
	
    # load image
	input_image = Image.open(os.path.join(input_dir, files)).convert("RGB")
	
	# resize image, keep aspect ratio
	h = input_image.size[0]
	w = input_image.size[1]
	ratio = h *1.0 / w
	if ratio > 1:
		h = load_size
		w = int(h*1.0/ratio)
	else:
		w = load_size
		h = int(w * ratio)
	input_image = input_image.resize((h, w), Image.BICUBIC)
	input_image = np.asarray(input_image)
	
	# RGB -> BGR
	input_image = input_image[:, :, [2, 1, 0]]
	input_image = transforms.ToTensor()(input_image).unsqueeze(0)
	
    # preprocess, (-1, 1)
	input_image = -1 + 2 * input_image 
	
	
    # forward
	output_image = model(input_image)
	output_image = output_image[0]
	# BGR -> RGB
	output_image = output_image[[2, 1, 0], :, :]
	# deprocess, (0, 1)
	output_image = output_image.data.cpu().float() * 0.5 + 0.5
	
	# save
	vutils.save_image(output_image, os.path.join(output_dir, files[:-4] + '_' + style + '.jpg'))