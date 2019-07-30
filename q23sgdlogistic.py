from UfileLoader import dataloader
from FNet import Net
from LSoftMax import SoftMax
from LCrossEntropy import crossentropy
from FSGD import SGD

dataset_train_data,dataset_train_labels,dataset_validate_data,dataset_validate_labels=dataloader()



logisticreg=Net()

D=1   #what data we are training on, from 0 to 2

print(dataset_train_data[D].shape)
print(dataset_train_labels[D].shape)



logisticreg.add_layer(SoftMax(dataset_train_data[D].shape[0],dataset_train_labels[D].shape[0]))
logisticreg.set_loss(crossentropy())
_,_,_,_= SGD(logisticreg,dataset_train_data[D],dataset_train_labels[D],dataset_validate_data[D],dataset_validate_labels[D],lr=0.001,momentum=0.8,batchsize=100,gamma=0.001,epochs=30,plot=True,plotProgress=False)