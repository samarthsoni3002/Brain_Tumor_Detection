import torch 
from utils import get_class_names
from model_builder import model_builder


def save_model(model: torch.nn.Module,model_path_name:str):
    
    model_path = model_path_name
    torch.save(model.state_dict(),model_path)
    

def load_model(model_path:str):
    
    class_names,_,_ = get_class_names("./brain_tumor_dataset/Training")
    
    model = model_builder(len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model