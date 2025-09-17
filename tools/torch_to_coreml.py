import torch
import coremltools as ct

# Load your TorchScript model
model_path = "rtmdet-nano_traced.pt"
traced_model = torch.jit.load(model_path)
traced_model.eval()

# Convert the PyTorch model to CoreML with simple normalization
coreml_model = ct.convert(
    traced_model,
    convert_to="neuralnetwork",
    inputs=[
        ct.ImageType(
            shape=(1, 3, 320, 320),
            scale=1.0 / 255.0 
        )
    ]
)

# Save the converted CoreML model
coreml_model.save("rtmdet-nano_Image.mlmodel")
print("Model successfully converted and saved as 'rtmdet-nano_Image.mlmodel'")