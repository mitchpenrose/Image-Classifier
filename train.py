import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import sys


class Train:
    arguments = None
    model = None
    device = None
    train_data = None
    trainloader = None
    validloader = None
    optimizer = None
    criterion = None
    
    def __init__(self, arguments):
        self.counter=0
        self.args=[]
        self.arguments = arguments
        self.build()
        self.load()
        self.train()
        self.save()
        
        
    def build(self):
        print("building model")
        if self.arguments.gpu:
            self.device = torch.device("cuda")
            print("using cuda")
        else:
            self.device = torch.device("cpu")
            print("using cpu")

        if self.arguments.arch == "vgg11":
            self.model = models.vgg11(pretrained=True)
        elif self.arguments.arch == "vgg13":
            self.model = models.vgg13(pretrained=True)
        elif self.arguments.arch == "vgg16":
            self.model = models.vgg16(pretrained=True)
        elif self.arguments.arch == "vgg19":
            self.model = models.vgg19(pretrained=True)
        elif self.arguments.arch == "resnet18":
            self.model = models.resnet18(pretrained=True)        
        elif self.arguments.arch == "resnet34":
            self.model = models.resnet34(pretrained=True)        
        elif self.arguments.arch == "resnet50":
            self.model = models.resnet50(pretrained=True)
        elif self.arguments.arch == "resnet101":
            self.model = models.resnet101(pretrained=True)
        elif self.arguments.arch == "resnet152":
            self.model = models.resnet152(pretrained=True)        

        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(nn.Linear(self.model.classifier[0].in_features, self.arguments.hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.2),
                                         nn.Linear(args.hidden_units, 102),
                                         nn.LogSoftmax(dim=1))
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.arguments.learning_rate)
        self.model.to(self.device)


    def load(self):
        print("loading data")
        train_dir = self.arguments.path + '/train'
        valid_dir = self.arguments.path + '/valid'
        test_dir = self.arguments.path + '/test'
        valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        self.train_data = datasets.ImageFolder(self.arguments.path + '/train', transform=train_transforms)
        valid_data = datasets.ImageFolder(self.arguments.path + '/valid', transform=valid_test_transforms)
        test_data = datasets.ImageFolder(self.arguments.path + '/test', transform=valid_test_transforms)

        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


    def train(self):
        print("training model")
        epochs = self.arguments.epochs
        steps = 0
        running_loss = 0
        print_every = 5
        for epoch in range(epochs):
            for inputs, labels in self.trainloader:
                steps += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                logps = self.model.forward(inputs)
                loss = self.criterion(logps, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.validloader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            logps = self.model.forward(inputs)
                            batch_loss = self.criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Valid loss: {test_loss/len(self.validloader):.3f}.. "
                          f"Valid accuracy: {accuracy/len(self.validloader):.3f}")
                    running_loss = 0
                    self.model.train()

    def save(self):
        print("saving model")
        self.model.class_to_idx = self.train_data.class_to_idx
        checkpoint = {
        'state_dict': self.model.state_dict(),
        'classifier': self.model.classifier,
        'epochs': self.arguments.epochs,
        'optimizer_state_dict': self.optimizer.state_dict(),
        'class_to_idx': self.model.class_to_idx,
        'model_type': self.arguments.arch}
        torch.save(checkpoint, self.arguments.save_dir + "checkpoint.pth")

        
parser = argparse.ArgumentParser()
parser.add_argument("path", metavar="N")
parser.add_argument("--save_dir", default="./")
parser.add_argument("--arch", default="vgg13")
parser.add_argument("--learning_rate", default=0.001)
parser.add_argument("--hidden_units", default=512)
parser.add_argument("--epochs", default=1)
parser.add_argument("--gpu", action='store_true')
args = parser.parse_args()
Train(args)