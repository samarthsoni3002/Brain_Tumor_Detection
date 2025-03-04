import torch 


def save_model(model: torch.nn.Module,model_path_name:str):
    
    model_path = model_path_name
    torch.save(model.state_dict(),model_path)