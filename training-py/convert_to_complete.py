from keras import models, layers, optimizers

model = models.Sequential()

# Convolutional Layers
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(100, 100, 3), activation='relu'))
# model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
# model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
# model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

# Dense Layers TODO: How many dense layers are needed? How many neurons per layer?
model.add(layers.Dense(256))
model.add(layers.Activation('relu'))
model.add(layers.Dense(256))
model.add(layers.Activation('relu'))

# Dropout helps prevent over fitting
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1))
# Sigmoid scales data for a binary output
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(amsgrad=True),  # TODO: Best optimizer?
              metrics=['accuracy'])

model.load_weights("AMSGrad-0.908.hdf5")
model.save("full-model.hdf5")