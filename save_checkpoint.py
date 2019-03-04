import torch

def save_checkpoint(arch, model, optimizer, epochs, filename="checkpoint.pth"):
    model.cpu()

    checkpoint = {'epochs': epochs,
                  'architecture': arch,
                  'classifier': model.classifier,
                  'optimizer_state': optimizer.state_dict,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, filename)