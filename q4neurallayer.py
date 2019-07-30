from FNet import Net
from LNeuralLayer import NeuralLayer
from LNeuralLayerT import NeuralLayerT
from TJacobianTest import JacobianTest
from TJacMVRT import JacMVRT
from TTransposeTest import TransposeTest

from UfileLoader import dataloader

dataset_train_data,dataset_train_labels,dataset_validate_data,dataset_validate_labels=dataloader()

softmaxtest=Net()

X=dataset_train_data[0]
#X=np.array([[1,2,1,2],[1,3,1,2]])

labels=dataset_train_labels[0]

jacobtest=NeuralLayer(X.shape[0],3)
jacobtestT=NeuralLayerT(X.shape[0],3)

#Test equivalent of function JacMVRT to backward function:
TransposeTest(jacobtest,X,JacMVRT)
TransposeTest(jacobtestT,X,JacMVRT)

#Jacobian test with JacMVRT:
for i in range(3):
    JacobianTest(jacobtest,X,JacMVRT,"q4Relu" + str(i))

for i in range(3):
    JacobianTest(jacobtestT,X,JacMVRT,"q4Tanh" + str(i))