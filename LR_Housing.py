import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt


def predict(X, w, y = None):

    m = X.shape[0] 
    y_hat = np.matmul(X,w)
    loss  = (0.5)*(1/m)*(np.linalg.norm(y_hat - y))
    y_new = (y * std_y) + mean_y
    y_hat_new = (y_hat * std_y) + mean_y
    risk  = np.mean(abs(y_hat_new - y_new))
    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val):
    N_train = X_train.shape[0]
    N_val   = X_val.shape[0]

    w = np.zeros([X_train.shape[1], 1])

    m = X_train.shape[0]
    losses_train = []
    risks_val   = []

    w_best    = None
    risk_best = 10000
    epoch_best= 0
    
    for epoch in range(MaxIter):

        loss_this_epoch = 0
        for b in range( int(np.ceil(N_train/batch_size)) ):
            
            X_batch = X_train[b*batch_size : (b+1)*batch_size]
            y_batch = y_train[b*batch_size : (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch


            # Mini-batch gradient descent
            A = np.matmul(np.matmul(X_batch.transpose(),X_batch), w)
            B = np.matmul(y_batch.transpose(), X_batch)
            Delta_J = (1/m)*(A - B)
            w = w - (alpha * Delta_J)
        
        tl = loss_this_epoch / m
        losses_train.append(tl)
        y_hat_epoch, loss_epoch, R = predict(X_val, w, y_val)
        risks_val.append(R)
        if risk_best > R:
            risk_best = R
            w_best = w
            epoch_best = epoch
    print("Best epoch = "+str(epoch_best))
    val = predict(X_val, w_best, y_val)[2]
    print("Best Validation Risk = " +str(val))
    
    # Return some variables as needed
    return w_best, losses_train, risks_val



############################
# Main code starts here
############################


X, y = datasets.load_boston(return_X_y=True)
y = y.reshape([-1, 1])
# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)



# Augment feature
X_ = np.concatenate( ( np.ones([X.shape[0],1]), X ), axis=1)
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y  = np.std(y)

y = (y - np.mean(y)) / np.std(y)



# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]

X_val   = X_[300:400]
y_val   = y[300:400]

X_test = X_[400:]
y_test = y[400:]

#####################
# setting

alpha   = 0.001     # learning rate
batch_size   = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = 0.0         # weight decay



w , TrainLoss , ValRisk = train(X_train, y_train, X_val, y_val)

# Perform test by the weights yielding the best validation performance
Y_hat , Loss , Risk = predict(X_test, w, y_test)
print("Test Risk = "+str(Risk))
# Report numbers and draw plots as required.
plt.plot(TrainLoss)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss over 100 Epochs')
plt.savefig("LR_TrainingLoss.png")
plt.show()


plt.plot(ValRisk)
plt.xlabel('Epochs')
plt.ylabel('Validation Risk')
plt.title('Validation Risk over 100 Epochs')
plt.savefig("LR__ValidationRisk.png")
plt.show()
