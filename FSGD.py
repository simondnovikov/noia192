import numpy as np
import matplotlib.pyplot as plt


def SGD(TrainedNet, data, labels, dataval, labelsval, lr, momentum, batchsize, gamma=0, epochs=20, plot=False,
        plotProgress=False):
    stat_train_acc = []
    stat_train_loss = []
    stat_val_acc = []
    stat_val_loss = []

    m = data.shape[1]
    mv = dataval.shape[1]
    if plotProgress:
        means = np.mean(data, axis=1)
        r = np.ptp(data, axis=1)
        X_n = np.random.random(data.shape)
        for dim in range(data.shape[0]):
            X_n[dim, :] = (X_n[dim, :] - 0.5) * r[dim] + means[dim]
    for iter in range(epochs):

        order = np.random.permutation(m)
        i = 0
        train_acc = 0
        train_loss = 0
        val_acc = 0
        val_loss = 0
        while i < m:
            batchdata = data[:, order[i:(i + batchsize)]]
            batchlabel = labels[:, order[i:(i + batchsize)]]
            i = i + batchsize

            loss, outlabel = TrainedNet.forward(batchdata, batchlabel)
            gradback = TrainedNet.backward()

            TrainedNet.update(lr, momentum, gamma)

            train_acc = train_acc + np.sum(outlabel == np.argmax(batchlabel, axis=0))
            train_loss = train_loss + loss

        stat_train_acc.append(train_acc / m)
        stat_train_loss.append(train_loss / m)

        lossv, outlabelv = TrainedNet.forward(dataval, labelsval)
        stat_val_acc.append(np.mean(outlabelv == labelsval))
        stat_val_loss.append(lossv / mv)
        print(np.max(gradback))
        if plotProgress:
            pmatnotneeded, outlabeltoplot = TrainedNet.forward(X_n)
            plt.scatter(X_n[0, :], X_n[1, :], c=outlabeltoplot)
            plt.show()
            print(outlabeltoplot)

    if plot:
        plt.plot(stat_train_acc, label="train")
        plt.plot(stat_val_acc, label="val")
        title = "Accuracy"
        plt.title(title)
        plt.legend()
        plt.show()

        plt.plot(stat_train_loss, label="train")
        plt.plot(stat_val_loss, label="val")
        title = "loss"
        plt.title(title)
        plt.legend()
        plt.show()
    return stat_train_acc, stat_val_acc, stat_train_loss, stat_val_loss