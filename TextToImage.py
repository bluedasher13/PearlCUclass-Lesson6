from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", use_safetensors=True
)

prompt = "cinematic photo of Godzilla eating sushi with a cat in a izakaya."
generator = torch.Generator(device="cpu").manual_seed(37)
image = pipeline(
    prompt,
    generator=generator,
    num_inference_steps=5,
).images[0]
image.save("image_of_squirrel_painting.png")