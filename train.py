from get_input_args import get_train_input_args
from get_data import *
from do_deep_learning import *
from check_accuracy_on_test import *
from save_checkpoint import *

from torch import nn
from torch import optim

def main():
    #Retrieve command line arguments from user as input from the user running the program from a terminal window
    in_arg = get_train_input_args()
    
    #Get and transform data
    data_dir = in_arg.data_dir
    train_data, trainloader, validloader, testloader = get_data(data_dir)
    
    #Build and train network
    
    #Get pretrained model and attach appropriate classifier
    if in_arg.arch.lower().strip() == 'vgg16':
        model = models.vgg16(pretrained=True)
        
        if in_arg.hidden_units:
            classifier = nn.Sequential(nn.Linear(25088,in_arg.hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(in_arg.hidden_units,102),
                                       nn.LogSoftmax(dim=1))
        else:
            classifier = nn.Sequential(nn.Linear(25088,4096),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(4096,4096),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(4096,4096),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(4096,102),
                                       nn.LogSoftmax(dim=1))
    elif in_arg.arch.lower().strip() == 'alexnet':
        model = models.alexnet(pretrained=True)
  
        if in_arg.hidden_units:
            classifier = nn.Sequential(nn.Dropout(p=0.5),
                                       nn.Linear(9216,in_arg.hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(in_arg.hidden_units,102),
                                       nn.LogSoftmax(dim=1))
        else:
            classifier = nn.Sequential(nn.Dropout(p=0.5),
                                       nn.Linear(9216,4096),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(4096,4096),
                                       nn.ReLU(),
                                       nn.Linear(4096,102),
                                       nn.ReLU(),
                                       nn.LogSoftmax(dim=1))
    else:
        print("Please choose model vgg16 or alexnet")
                                       
    #Set up classifier for chosen pretrained model
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = in_arg.learning_rate)
    
    #Train model and print loss using the train and validation datasets
    check = do_deep_learning(model, trainloader, in_arg.epochs, 25, criterion, optimizer, in_arg.gpu)
    checkval = do_deep_learning(model, validloader, in_arg.epochs, 25, criterion, optimizer, in_arg.gpu)
    
    #Test and print network's accuracy:
    check_accuracy_on_test(model, testloader, in_arg.gpu)
    
    #Save checkpoint
    if in_arg.save_dir:
        filename = in_arg.save_dir + 'checkpoint.pth'
    else:
        filename = 'checkpoint.pth'
    model.class_to_idx = train_data.class_to_idx
    save_checkpoint(in_arg.arch.lower().strip(), model, optimizer, in_arg.epochs, filename)

# Call to main function to run the program
if __name__ == "__main__":
    main()