from UfileLoader import dataloader
from FNet import Net
from LSoftMax import SoftMax
from LCrossEntropy import crossentropy
from TGradientTestX import GradientTestX
from TGradientTestW import  GradientTestW


dataset_train_data,dataset_train_labels,dataset_validate_data,dataset_validate_labels=dataloader()

softmaxtest=Net()

X=dataset_train_data[0]
#X=np.array([[1,2,1,2],[1,3,1,2]])

labels=dataset_train_labels[0]
#labels=np.array([[1,0,1,0],[0,1,0,1]])

#weights = np.array([[1,2],[2,1],[1,1]])


softmaxtest.add_layer(SoftMax(X.shape[0],labels.shape[0]))
softmaxtest.set_loss(crossentropy())


#softmaxtest.add_layer(Layer(X.shape[0],labels.shape[0]))
#softmaxtest.set_loss(LossFunction())

for i in range(3):
    GradientTestX(softmaxtest,X,labels,"q1X" + str(i))
for i in range(3):
    GradientTestW(softmaxtest,X,labels,"q1W" + str(i))