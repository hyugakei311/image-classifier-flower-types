import torch

def check_accuracy_on_test(model, testloader, gpu=False):
    correct = 0
    total = 0
    
    if gpu:
        model.to('cuda')
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' % (100 * correct / total))