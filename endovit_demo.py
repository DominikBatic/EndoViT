import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from torch import nn
from huggingface_hub import snapshot_download


def process_single_image(image_path, input_size=224, dataset_mean=[0.3464, 0.2280, 0.2228], dataset_std=[0.2520, 0.2128, 0.2093]):
    # Define the transformations
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    # Open the image
    image = Image.open(image_path).convert('RGB')

    # Apply the transformations
    processed_image = transform(image)

    return processed_image


def load_model_from_huggingface(repo_id, model_filename):
    # Download model files
    model_path = snapshot_download(repo_id=repo_id, revision="main")
    model_weights_path = Path(model_path) / model_filename

    # Load model weights
    model_weights = torch.load(model_weights_path)['model']

    # Define the model (ensure this matches your model's architecture)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).eval()

    # Load the weights into the model
    loading = model.load_state_dict(model_weights, strict=False)

    return model, loading


image_paths = sorted(Path('demo_images').glob('*.png'))  # TODO replace with image paths
images = torch.stack([process_single_image(image_path) for image_path in image_paths])

device = "cuda"
dtype = torch.float16
model, loading_info = load_model_from_huggingface("egeozsoy/EndoViT", "pytorch_model.bin")
model = model.to(device, dtype)
print(loading_info)
output = model.forward_features(images.to(device, dtype))
print(output.shape)
