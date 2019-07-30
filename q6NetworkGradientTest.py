from UfileLoader import dataloader
from FNet import Net
from LSoftMax import SoftMax
from LCrossEntropy import crossentropy
from TGradientTestX import GradientTestX
from TGradientTestW import  GradientTestW
from LNeuralLayer import NeuralLayer
from LNeuralLayerT import NeuralLayerT

dataset_train_data,dataset_train_labels,dataset_validate_data,dataset_validate_labels=dataloader()

LargeNet=Net()

X=dataset_train_data[0]
#X=np.array([[1,2,1,2],[1,3,1,2]])

labels=dataset_train_labels[0]
#labels=np.array([[1,0,1,0],[0,1,0,1]])

#weights = np.array([[1,2],[2,1],[1,1]])

LargeNet.add_layer(NeuralLayerT(X.shape[0],10))
LargeNet.add_layer(NeuralLayerT(10,20))
LargeNet.add_layer(SoftMax(20,labels.shape[0]))
LargeNet.set_loss(crossentropy())


for i in range(3):
    GradientTestX(LargeNet,X,labels,"q6X" + str(i))
for i in range(3):
    GradientTestW(LargeNet,X,labels,"q6W" + str(i))