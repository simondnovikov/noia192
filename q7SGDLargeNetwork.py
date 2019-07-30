from UfileLoader import dataloader
from FNet import Net
from LSoftMax import SoftMax
from LCrossEntropy import crossentropy
from TGradientTestX import GradientTestX
from TGradientTestW import  GradientTestW
from LNeuralLayer import NeuralLayer
from LNeuralLayerT import NeuralLayerT
from FSGD import SGD

dataset_train_data,dataset_train_labels,dataset_validate_data,dataset_validate_labels=dataloader()

LargeNet=Net()
D=2
X=dataset_train_data[D]
#X=np.array([[1,2,1,2],[1,3,1,2]])

labels=dataset_train_labels[D]
#labels=np.array([[1,0,1,0],[0,1,0,1]])

#weights = np.array([[1,2],[2,1],[1,1]])

LargeNet.add_layer(NeuralLayerT(X.shape[0],10))
LargeNet.add_layer(NeuralLayerT(10,10))
LargeNet.add_layer(NeuralLayerT(10,10))
LargeNet.add_layer(NeuralLayerT(10,10))
LargeNet.add_layer(NeuralLayerT(10,10))
LargeNet.add_layer(NeuralLayerT(10,10))
LargeNet.add_layer(NeuralLayerT(10,10))
LargeNet.add_layer(SoftMax(10,labels.shape[0]))
LargeNet.set_loss(crossentropy())

_,_,_,_= SGD(LargeNet,dataset_train_data[D],dataset_train_labels[D],dataset_validate_data[D],dataset_validate_labels[D],lr=0.03,momentum=0,batchsize=50,gamma=0,epochs=200,plot=True,plotProgress=False,f="q7training")