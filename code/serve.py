import io
import base64
import logging
import numpy as np
from PIL import Image
import torch, os, json
from model import Transformer
# import torchvision.utils as vutils
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

def input_fn(request_body, request_content_type):
    logger.info('Receiving input.')
    if request_content_type == 'application/json':
        logger.info(f'request_body : {request_body}')
        data = json.loads(request_body)
        # input_tensor = torch.tensor(data)
        logger.info(f"Passed input : {data['body']}")
        return data['body']
    else:
        logger.exception(f"Unsupported content type: {request_content_type}")
        # raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    logger.info('Generating cartoonization.')
	
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
    logger.info(f"input : {input_data}")
    imgdata = base64.b64decode(input_data)
    logger.info(f"input2 : {imgdata}")
    input_image = Image.open(io.BytesIO(imgdata)).convert("RGB")
    logger.info('Complete base64 to img.')
    
    # ext = os.path.splitext(input_data)[1]
    # if ext not in ['.jpg', '.png']:
    #     logger.exception('Invalid input type.')
    #     return None
    # logger.info('Valid datatype')

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
        input_image = input_image.to(device)
        cartoonized = model(input_image)

    output_image = cartoonized[0]
	# BGR -> RGB
    output_image = output_image[[2, 1, 0], :, :]
	# deprocess, (0, 1)
    output_image = output_image.data.cpu().float() * 0.5 + 0.5

    logger.info('Complete prediction')
    return output_image

def output_fn(prediction, response_content_type):
    logger.info(f'Generating output.')
    if response_content_type == 'application/json':
        transform = transforms.ToPILImage("RGB")
        pil_image = transform(prediction)

        # Create in-memory file from the PIL image
        in_mem_file = io.BytesIO()
        pil_image.save(in_mem_file, format = "PNG")

        # Read the bytes
        in_mem_file.seek(0)
        image_bytes = in_mem_file.read()

        # Encode the bytes to base64
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        logger.info(f'Output generated')
        return base64_string
    else:
        logger.exception(f"Unsupported content type: {response_content_type}")
        # raise ValueError(f"Unsupported content type: {response_content_type}")