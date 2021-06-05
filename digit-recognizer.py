import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image


######################################################################
# OneLayerNetwork
######################################################################

class OneLayerNetwork(torch.nn.Module):
    def __init__(self):
        super(OneLayerNetwork, self).__init__()

        ### ========== TODO : START ========== ###
        ### part d: implement OneLayerNetwork with torch.nn.Linear
        self.layer = torch.nn.Linear(784, 3)
        ### ========== TODO : END ========== ###

    def forward(self, x):
        # x.shape = (n_batch, n_features)

        ### ========== TODO : START ========== ###
        ### part d: implement the foward function
        x = x.view(x.size(0), -1)
        L1_out = self.layer(x)
        outputs = L1_out
        ### ========== TODO : END ========== ###
        return outputs
      
      
######################################################################
# TwoLayerNetwork
######################################################################

class TwoLayerNetwork(torch.nn.Module):
    def __init__(self):
        super(TwoLayerNetwork, self).__init__()
        ### ========== TODO : START ========== ###
        ### part g: implement TwoLayerNetwork with torch.nn.Linear
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(784, 400), torch.nn.ReLU(), torch.nn.Sigmoid(), torch.nn.Linear(400, 3))        
        ### ========== TODO : END ========== ###

    def forward(self, x):
        # x.shape = (n_batch, n_features)

        ### ========== TODO : START ========== ###
        ### part g: implement the foward function
        x = x.view(x.size(0), -1)
        L2_out = self.layer1(x)
        
        outputs = L2_out
        ### ========== TODO : END ========== ###
        return outputs

      
# load data from csv
# X.shape = (n_examples, n_features), y.shape = (n_examples, )
def load_data(filename):
    data = np.loadtxt(filename)
    y = data[:, 0].astype(int)
    X = data[:, 1:].astype(np.float32) / 255
    return X, y
  
 
# plot one example
# x.shape = (features, )
def plot_img(x):
    x = x.reshape(28, 28)
    img = Image.fromarray(x*255)
    plt.figure()
    plt.imshow(img)
    return
 

def evaluate_loss(model, criterion, dataloader):
    model.eval()
    total_loss = 0.0
    for batch_X, batch_y in dataloader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        total_loss += loss.item()
        
    return total_loss / len(dataloader)
  
  
 def evaluate_acc(model, dataloader):
    model.eval()
    total_acc = 0.0
    for batch_X, batch_y in dataloader:
        outputs = model(batch_X)
        predictions = torch.argmax(outputs, dim=1)
        total_acc += (predictions==batch_y).sum()
        
    return total_acc / len(dataloader.dataset)
  
  
def train(model, criterion, optimizer, train_loader, valid_loader):
    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    for epoch in range(1, 31):
        model.train()
        for batch_X, batch_y in train_loader:
            ### ========== TODO : START ========== ###
            ### part f: implement the training process
            #forward pass, initializing gradients to zero,
            # computing loss, loss.backward, updating model parameters
            images = torch.autograd.Variable(batch_X)
            labels = torch.autograd.Variable(batch_y)

            optimizer.zero_grad()

            logits = model(images)
            #compute loss
            loss = criterion(logits, labels)

            #backpropagation
            loss.backward()
            #update
            optimizer.step()
            ### ========== TODO : END ========== ###
            
        train_loss = evaluate_loss(model, criterion, train_loader)
        valid_loss = evaluate_loss(model, criterion, valid_loader)
        train_acc = evaluate_acc(model, train_loader)
        valid_acc = evaluate_acc(model, valid_loader)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        print(f"| epoch {epoch:2d} | train loss {train_loss:.6f} | train acc {train_acc:.6f} | valid loss {valid_loss:.6f} | valid acc {valid_acc:.6f} |")

    return train_loss_list, valid_loss_list, train_acc_list, valid_acc_list
  
  
######################################################################
# main
######################################################################

def main():

    # fix random seed
    np.random.seed(0)
    torch.manual_seed(0)

    # load data with correct file path

    ### ========== TODO : START ========== ###
    data_directory_path =  "/content/drive/My Drive/CSM146PS3"
    ### ========== TODO : END ========== ###

    # X.shape = (n_examples, n_features)
    # y.shape = (n_examples, )
    X_train, y_train = load_data(os.path.join(data_directory_path, "ps3_train.csv"))
    X_valid, y_valid = load_data(os.path.join(data_directory_path, "ps3_valid.csv"))
    X_test, y_test = load_data(os.path.join(data_directory_path, "ps3_test.csv"))

    ### ========== TODO : START ========== ###
    ### part a: print out three training images with different labels
    visited_labels = []
    while len(visited_labels) < 3:
      x = np.random.randint(0, X_train.shape[0])
      if y_train[x] not in visited_labels:
          visited_labels.append(y_train[x])
          plot_img(X_train[x])
    ### ========== TODO : END ========== ###

    print("Data preparation...")

    ### ========== TODO : START ========== ###
    ### part b: convert numpy arrays to tensors
    tX_train = torch.Tensor(X_train)
    ty_train = torch.Tensor(y_train).type(torch.LongTensor)
    tX_valid = torch.Tensor(X_valid)
    ty_valid = torch.Tensor(y_valid).type(torch.LongTensor)
    tX_test = torch.Tensor(X_test)
    ty_test = torch.Tensor(y_test).type(torch.LongTensor)
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    ### part c: prepare dataloaders for training, validation, and testing
    ###         we expect to get a batch of pairs (x_n, y_n) from the dataloader
    
    train_dataset = torch.utils.data.TensorDataset(tX_train, ty_train)
    valid_dataset = torch.utils.data.TensorDataset(tX_valid, ty_valid)
    test_dataset = torch.utils.data.TensorDataset(tX_test, ty_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    ### part e: prepare OneLayerNetwork, criterion, and optimizer
    model_one = OneLayerNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_one.parameters(), lr=0.0005)
    ### ========== TODO : END ========== ###

    print("Start training OneLayerNetwork...")
    results_one = train(model_one, criterion, optimizer, train_loader, valid_loader)
    print("Done!")

    ### ========== TODO : START ========== ###
    ### part h: prepare TwoLayerNetwork, criterion, and optimizer
    model_two = TwoLayerNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_two.parameters(), lr=0.0005)
    ### ========== TODO : END ========== ###

    print("Start training TwoLayerNetwork...")
    results_two = train(model_two, criterion, optimizer, train_loader, valid_loader)
    print("Done!")

    one_train_loss, one_valid_loss, one_train_acc, one_valid_acc = results_one
    two_train_loss, two_valid_loss, two_train_acc, two_valid_acc = results_two

    ### ========== TODO : START ========== ###
    ### part i: generate a plot to comare one_train_loss, one_valid_loss, two_train_loss, two_valid_loss
    plt.figure(figsize=(12.8,9.6))
    x = [p for p in range(1,31)]
    plt.plot(x, one_train_loss, label="one_train_loss", marker='x')
    plt.plot(x, one_valid_loss, label="one_valid_loss", marker='x')
    plt.plot(x, two_train_loss, label="two_train_loss", marker='x')
    plt.plot(x, two_valid_loss, label="two_valid_loss", marker='x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.show

    plt.savefig ('loss.pdf')
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    ### part j: generate a plot to compare one_train_acc, one_valid_acc, two_train_acc, two_valid_acc
    plt.figure(figsize=(12.8,9.6))
    x = [p for p in range(1,31)]
    plt.plot(x, one_train_acc, label="one_train_acc", marker='x')
    plt.plot(x, one_valid_acc, label="one_valid_acc", marker='x')
    plt.plot(x, two_train_acc, label="two_train_acc", marker='x')
    plt.plot(x, two_valid_acc, label="two_valid_acc", marker='x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend()
    plt.show

    plt.savefig ('accuracy.pdf')
    ### ========== TODO : END ========== ##

    ### ========== TODO : START ========== ###
    ### part k: calculate the test accuracy
    print("OneLayerNetwork test accuracy")
    print(evaluate_acc(model_one, test_loader))
    print("TwoLayerNetwork test accuracy")
    print(evaluate_acc(model_two, test_loader))
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    ### part l: replace the SGD optimizer with the Adam optimizer and do the experiments again
    print("Testing with the Adam optimizer")
    model_one_Adam = OneLayerNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_Adam = torch.optim.Adam(model_one_Adam.parameters(), lr=0.0005)
    results_one_Adam = train(model_one_Adam, criterion, optimizer_Adam, train_loader, valid_loader)
    print("Training the TwoLayerNetwork")
    model_two_Adam = TwoLayerNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_Adam = torch.optim.Adam(model_two_Adam.parameters(), lr=0.0005)
    results_two_Adam = train(model_two_Adam, criterion, optimizer_Adam, train_loader, valid_loader)

    one_train_loss, one_valid_loss, one_train_acc, one_valid_acc = results_one_Adam
    two_train_loss, two_valid_loss, two_train_acc, two_valid_acc = results_two_Adam

    ### generate a plot to compare one_train_loss, one_valid_loss, two_train_loss, two_valid_loss
    plt.figure(figsize=(12.8,9.6))
    x = [p for p in range(1,31)]
    plt.plot(x, one_train_loss, label="one_train_loss", marker='x')
    plt.plot(x, one_valid_loss, label="one_valid_loss", marker='x')
    plt.plot(x, two_train_loss, label="two_train_loss", marker='x')
    plt.plot(x, two_valid_loss, label="two_valid_loss", marker='x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.show

    plt.savefig ('loss_Adam.pdf')

    ### generate a plot to compare one_train_acc, one_valid_acc, two_train_acc, two_valid_acc
    plt.figure(figsize=(12.8,9.6))
    x = [p for p in range(1,31)]
    plt.plot(x, one_train_acc, label="one_train_acc", marker='x')
    plt.plot(x, one_valid_acc, label="one_valid_acc", marker='x')
    plt.plot(x, two_train_acc, label="two_train_acc", marker='x')
    plt.plot(x, two_valid_acc, label="two_valid_acc", marker='x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend()
    plt.show

    plt.savefig ('accuracy_Adam.pdf')

    ### calculate the test accuracy
    print("OneLayerNetwork test accuracy")
    print(evaluate_acc(model_one_Adam, test_loader))
    print("TwoLayerNetwork test accuracy")
    print(evaluate_acc(model_two_Adam, test_loader))
    ### ========== TODO : END ========== ###



if __name__ == "__main__":
    main() 
