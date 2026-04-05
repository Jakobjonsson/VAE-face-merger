import os

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

cuda_avaliable = torch.cuda.is_available()


class VAE_FaceMerger():
    def __init__(self, image_path1, image_path2):
        self.vae_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        if not cuda_avaliable:
            print("CUDA not available, using CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
        self.vae_model.to(self.device)
        self.vae_model.eval()
        self.image1 = Image.open(image_path1).convert("RGB")
        self.image2 = Image.open(image_path2).convert("RGB")
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        self.z_mid = None
        self.merged_face = None
        self.image = None

    def preprocess(self, pil_image):
        return self.transform(pil_image).unsqueeze(0).to(self.device)
    
    def prerocess_images(self):
        if (self.image1 and self.image2):
            self.image1_tensor = self.preprocess(self.image1)
            self.image2_tensor = self.preprocess(self.image2)
        else:
            print("Requires input images. Use self.load_images()")

    def encode(self):

        if isinstance(self.image1_tensor, torch.Tensor) and isinstance(self.image2_tensor, torch.Tensor):
            with torch.no_grad():
                self.z1 = self.vae_model.encode(self.image1_tensor).latent_dist.sample() * 0.18215
                self.z2 = self.vae_model.encode(self.image2_tensor).latent_dist.sample() * 0.18215
        else:
            print("Requires . Use self.preprocess_images()")

    def linear_interpolation(self, t=0.5):
        if isinstance(self.z1, torch.Tensor) and isinstance(self.z2, torch.Tensor):
            self.z_mid = (1 - t) * self.z1 + t * self.z2
        else:
            print("Requires encoded images. Encode with self.encode()")
    
    def decode_z(self):
        if isinstance(self.z_mid, torch.Tensor):
                    
            with torch.no_grad():
                self.merged_face = self.vae_model.decode(self.z_mid / 0.18215).sample
        else:
            print("Requires self.z_mid. Use self.spherical_linear_interpolation()")

    
    def to_image(self):
        if isinstance(self.merged_face, torch.Tensor):
            img = (self.merged_face.clamp(-1, 1) + 1) / 2
            img = img.cpu().permute(0, 2, 3, 1).squeeze()
            self.image = Image.fromarray((img.numpy() * 255).astype("uint8"))
        else:
            print("Requires self.merged_face. Use self.decode_z() after interpolation.")
        
    def merge_faces(self):
        try:
            print("Preprocessing images...")
            self.prerocess_images()
            print("Encoding images...")
            self.encode()
            print("Linear interpolation...")
            self.linear_interpolation()
            print("Decoding z...")
            self.decode_z()
            print("Converting to image...")
            self.to_image()
            print("Merging faces...")
            return self.image

        except Exception as e:
            print(e)
            print("Failed to merge faces. Check if the images are valid.")
            return None

if __name__ == '__main__':
    fm = VAE_FaceMerger("ulf_zoom.jpg", "karin_zoom.jpg")
    output_image = fm.merge_faces()
    output_image.show()
    save_path = "output_face.png"
    output_image.save(save_path)