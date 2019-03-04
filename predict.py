from get_input_args import get_predict_input_args
from load_checkpoint import *
from get_cat_to_name import *
from get_prediction import *
from display_prediction import *

def main():
    #Retrieve command line arguments from user as input from the user running the program from a terminal window
    in_arg = get_predict_input_args()
    
    #Get categories to name mapping
    cat_to_name = get_cat_to_name(in_arg.category_names)
    
    #Load checkpoint
    model, epochs, optimizer_state = load_checkpoint(in_arg.checkpoint)
    
    #Return top K predictions
    top_probs, top_classes = get_prediction(in_arg.img_dir, model, in_arg.topk, in_arg.gpu)
    
    #Display flower name and class probability
    display_prediction(in_arg.img_dir, cat_to_name, top_probs, top_classes)
    
# Call to main function to run the program
if __name__ == "__main__":
    main()