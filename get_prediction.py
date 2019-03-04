from process_image import *
import torch
import numpy

def get_prediction(image_path, model, topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(Image.open(image_path))
    img = torch.from_numpy(image).type(torch.FloatTensor) 
    img.unsqueeze_(0)
    
    model = model.eval()
    
    if gpu:
        model.to('cuda')
    
    model.type(torch.FloatTensor)
    results = torch.exp(model(img))
    top_probs, top_classes = results.topk(topk)
    top_probs = top_probs.detach().numpy().squeeze().tolist()
    top_classes = top_classes.detach().numpy().squeeze().tolist()
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    top_classes = [idx_to_class[i] for i in top_classes]
    
    return top_probs, top_classes