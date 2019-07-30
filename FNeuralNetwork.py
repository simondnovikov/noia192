from FNet import Net
from LSoftMax import SoftMax
from LCrossEntropy import crossentropy

def NeuralNetwork(chan_in, channel_sizes, classes, layers):
    net = Net()
    chan = chan_in
    for i in range(len(channel_sizes)):
        net.add_layer(layers[i](chan, channel_sizes[i]))
        chan = channel_sizes[i]

    net.add_layer(SoftMax(chan, classes))
    net.set_loss(crossentropy())
    return net