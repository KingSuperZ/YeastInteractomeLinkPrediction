""" Link Prediction in Graph Neural Networks

Link prediction looks at an edge list of nodes to predict whether certain nodes interact with each other

This code first takes data looking at the yeast interactome from the Harvard page on the CCSB Interactome. Then, before the training can be done, an edge list 
is created using the data and some rudimentary features based on the number of unique proteins, and, for the handmade model, the edge list is converted into an 
adjacency matrix. Next, two GCN models are created with one being the handmade model created from scratch and the other being a premade model from 
PyTorch Geometric. Within both of the models the data is being processed through three different layers where they are being converted into different numbers
of embedding per node. It starts with 905 embeddings and is then changed into 35, 32, and 37 embeddings per node across the three different layers. Then, the 
embeddings and edge list are used, for both the positive and negative edge list, to try and predict the existence of a link between two nodes. These values are 
then stored as binary values and is then checked in its accuracy by calculating the error. This process is repeated over 2000 epochs until error reaches as low 
as possible. Finally, the code is tested for accuracy to see how well it is able to predict the links. At the end, the model is able to almost perfectly predict
the existence of positive edges but only gets it right 60% of the time when it comes to negative edges.
"""

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm

## DATA ##
# Link to Data: http://interactome.dfci.harvard.edu/S_cerevisiae/download/CCSB-Y2H.txt
df = pd.read_csv("CCSB-Y2H.txt", sep = "\t", header = None, usecols = [0,1], names = ["Source","Destination"])
DF = df.apply(lambda col: col.astype('category').cat.codes) # Converts all of the values in the yeast interactome to integer values
print(DF)

numUniqueSource = len(pd.unique(DF["Source"])) # 905
numUniqueDest = len(pd.unique(DF["Destination"])) # 639
numUniqueProteins = len(DF.stack().unique()) # The same as numUniqueSource since the source contained all of the unique proteins

x = torch.eye(numUniqueProteins) # Creates the dummy features needed for the algorithm to run. Tried to use networkx to create some but it didn't improve accuracy
edge_index = torch.tensor(DF.values.T, dtype = torch.int64) # dataframe is reformatted to fit the COO format and be used as a tensor in the model
data = Data(x = x, edge_index = edge_index) # contains the dummy features and edges
dataAdj = to_dense_adj(data.edge_index)[0] # adjancency matrix of the edges (Used for Handmade Model)

## MODELS ##

# Prebuilt Model
class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_layer1, dim_layer2, dim_layer3):
        super().__init__()
        self.conv1 = GCNConv(dim_in, dim_layer1)
        self.conv2 = GCNConv(dim_layer1, dim_layer2)
        self.conv3 = GCNConv(dim_layer2, dim_layer3)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x) # Apply activation function
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        return x

# Handmade Model
class myGCNLayer(nn.Module):
  def __init__(self, dim_in, dim_out):
    super().__init__()
    self.lin1 = nn.Linear(dim_in, dim_out) # mainly addition and mutliplication
  def forward(self, x, adj):
    x = self.lin1(x) # each node is expanded from 1 feature to 16 node embeddings per node. The number 16 comes from the variable dim_hidden
    x = adj @ x
    return x

class myGCN(nn.Module):
  def __init__(self, dim_in, dim_layer1, dim_layer2, dim_layer3):
    super().__init__()
    self.layer1 = myGCNLayer(dim_in, dim_layer1)
    self.layer2 = myGCNLayer(dim_layer1, dim_layer2)
    self.layer3 = myGCNLayer(dim_layer2, dim_layer3)
  def forward(self, x, adj):
    x = self.layer1(x, adj)
    x = self.layer2(x, adj)
    x = self.layer3(x, adj)
    return x

numlayer1 = 35
numlayer2 = 32
numlayer3 = 37
model = GCN(data.num_features, numlayer1, numlayer2, numlayer3) # For Prebuilt Model
#model = myGCN(data.num_features, numlayer1, numlayer2, numlayer3) # For Handmade Model

## TRAINING ##
def link_predict(embedding, edgeindex):
  """This function tries to predict the existence of an edge through the dot product of the source and destination embedding of the edge being checked. Then, it 
  returns a value between 0 to 1 with values closer to one leaning towards there being an edge while values closer to 0 lean towards there not being an edge 
  """
  source = edgeindex[0]
  destination = edgeindex[1]
  return torch.sigmoid((embedding[source]*embedding[destination]).sum(axis = 1))
lossList = []
numepochs = 2000
lr = 0.01
ones = torch.ones(len(edge_index[0])) # Creates a tensor of ones that is the same length as the edge list
zeros = torch.zeros(len(edge_index[0])) # Creates a tensor of ones that is the same length as the edge list
y = torch.cat([ones,zeros]) # Combines the two tensors above to represent what the actual answer the algorithm should get as a result of running the prediction
lossfn = nn.BCELoss() # The BCE Loss is widely used for binary classification. This algorithm is binary classification since decides whether there is an edge or not
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in tqdm(range(numepochs)):
    embedding = model(data) # For Prebuilt Model
    #embedding = model(data.x, dataAdj) # For Handmade Mode
    
    # Next four lines run the function to try and predict whether a link exists between two nodes or not
    poslinkprediction = link_predict(embedding,edge_index) # Tries to predict the existence of a positive edge
    negedgeindex = torch.tensor(negative_sampling(edge_index)) # Is a list of negative edges
    neglinkprediction = link_predict(embedding,negedgeindex) # Treis to predict the existence of a negative edge
    linkpred = torch.cat([poslinkprediction,neglinkprediction]) # Combines the two predictions into one tensor
    
    # Standard PyTorch code for most models
    loss = lossfn(linkpred,y)
    loss.backward() # Computes the derivative/slope of the error
    optimizer.step() # Takes the step towards the local min using the lr
    optimizer.zero_grad() # Resets the slope calculation in order for change to occur
    lossList.append(loss.detach())
plt.plot(lossList)

## TESTING ##
testedge_index = torch.tensor(negative_sampling(edge_index)) # Creates a test tensor of negative edges
print(embedding) 
a = link_predict(embedding, edge_index) # Does one more prediction on the positive edges so the accuracy can be tested
print(a.round().sum()/len(a)) # Generates a number representing how accurate the prediction above is
b = link_predict(embedding, testedge_index) # Does one more prediction using the test negative edges so the accuracy can be tested
print(1- (b.round().sum()/len(b))) # Generates a number representing how accurate the prediction above is
