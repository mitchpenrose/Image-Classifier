import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import sys
from PIL import Image
import json


class Predict:
    arguments = None
    cat_to_name = {"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}
    
    
    def __init__(self, arguments):
        self.counter=0
        self.args=[]
        self.arguments = arguments
        if arguments.category_names != "":
            f = open(arguments.category_names, "r")
            self.cat_to_name = json.loads(f.read())
        model = self.load_checkpoint()
        probability, flowerKeys = self.predict(arguments.path, model, int(arguments.top_k))
        print(list(map(lambda x: self.cat_to_name[x], flowerKeys)), probability)
        
    def load_checkpoint(self):
        checkpoint = torch.load(self.arguments.checkpoint)
        model = None
        if checkpoint["model_type"] == "vgg11":
            model = models.vgg11(pretrained=True)
        elif checkpoint["model_type"] == "vgg13":
            model = models.vgg13(pretrained=True)
        elif checkpoint["model_type"] == "vgg16":
            model = models.vgg16(pretrained=True)
        elif checkpoint["model_type"] == "vgg19":
            model = models.vgg19(pretrained=True)
        elif checkpoint["model_type"] == "resnet18":
            model = models.resnet18(pretrained=True)        
        elif checkpoint["model_type"] == "resnet34":
            model = models.resnet34(pretrained=True)        
        elif checkpoint["model_type"] == "resnet50":
            model = models.resnet50(pretrained=True)
        elif checkpoint["model_type"] == "resnet101":
            model = models.resnet101(pretrained=True)
        elif checkpoint["model_type"] == "resnet152":
            model = models.resnet152(pretrained=True)  

        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx'] 
        return model
    
    
    def process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        width, height = image.size
        ratio = 0
        if width <= height:
            ratio = width / 256
        else:
            ratio = height / 256
        width = int(width / ratio)
        height = int(height / ratio)
        image = image.resize(size=(width, height))
        left = (width - 224)/2
        top = (height - 224)/2
        right = (width + 224)/2
        bottom = (height + 224)/2
        image = image.crop((left, top, right, bottom))
        normalized = np.array(image)/255
        normalized = (normalized - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
        normalized = np.transpose(normalized, (2,0,1))
        return torch.from_numpy(normalized).type(torch.FloatTensor)
    
        
    def predict(self, image_path, model, topk=1):
        device = None
        if self.arguments.gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        model.to(device)
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        npImage = self.process_image(Image.open(image_path))
        npImage = npImage.unsqueeze(0)
        npImage = npImage.to(device)
        model.eval()
        with torch.no_grad():
            logps = model.forward(npImage)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(topk, dim=1)

            probs = map(lambda x: x.item(), top_p[0])
            probs = list(probs)

            classs = map(lambda x: x.item(), top_class[0])
            classs = list(classs)

            classIndex = []
            for c in classs:
                for key, value in model.class_to_idx.items():
                    if value == c:
                        classIndex.append(key)

            return probs, classIndex

parser = argparse.ArgumentParser()
parser.add_argument("path", metavar="N")
parser.add_argument("checkpoint", metavar="N")
parser.add_argument("--top_k", default=1)
parser.add_argument("--category_names", default="")
parser.add_argument("--gpu", action='store_true')
args = parser.parse_args()
Predict(args)