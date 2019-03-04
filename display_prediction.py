from process_image import *

def display_prediction(image_path, cat_to_name, top_probs, top_classes):
    #Get true name
    flower_num = image_path.split('/')[2]
    true_name = cat_to_name[flower_num]
    #Get name from classes
    class_to_name = [cat_to_name[cl] for cl in top_classes]
    #Display result
    print("True name: ", true_name)
    for i in range(len(top_probs)):
        print(class_to_name[i].ljust(20,' '), top_probs[i])