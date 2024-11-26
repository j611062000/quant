from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.models import load_model as keras_load_model

def dump_model(model: Sequential, model_path: str):
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
    
    
def load_model(model_path: str) -> Sequential:
    return keras_load_model(model_path)