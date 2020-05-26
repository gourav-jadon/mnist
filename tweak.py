from keras.datasets import mnist
dataset = mnist.load_data('mymnist.db')
train , test = dataset
X_train , y_train = train
X_test , y_test = test
X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)
X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=2000, input_dim=28*28, activation='relu'))
model.add(Dense(units=1024, activation='relu',))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=400, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=10,batch_size=32, activation='softmax'))
model.summary()
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )
h = model.fit(X_train, y_train,batch_size=256, epochs=3)
model.save("mnistmodel.h5")
print("model trained successfully")
prediction = model.evaluate(X_test,y_test)
with open("accuracy.txt","w") as out_file:
    
        out_string = ""
        out_string += str(prediction[1])
        out_file.write(out_string)
print("file created successfully")