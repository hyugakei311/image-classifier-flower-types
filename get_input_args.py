import argparse

def get_train_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='flowers/', help='data directory')
    parser.add_argument('--save_dir', type=str, help='directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='the CNN model architecture: vgg16 or alexnet')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='set learning rate')
    parser.add_argument('--hidden_units', type=int, help='set number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='set number of epochs')
    parser.add_argument('--gpu', action='store_true', help='set GPU for training')

    return parser.parse_args()

def get_predict_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_dir', type=str, default='flowers/test/1/image_06743.jpg', help='path to image')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='directory to saved checkpoint')
    parser.add_argument('--topk', type=int, default=5, help='return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mappping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='set GPU for inference')

    return parser.parse_args()