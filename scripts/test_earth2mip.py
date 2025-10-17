from earth2mip.networks import get_model

MODEL_PATH = "/your/path/to/fcnv2_params"
model = get_model(f"file://{MODEL_PATH}")
print("Successfully obtained model!")