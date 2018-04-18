import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

class model1:
	
	def __init__(self, lr, decay, epoch):
		self.lr = lr
		self.decay = decay
		self.epoch = epoch

	def __graph__(self):
		self.model = Sequential()
		self.model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(48,48,1)))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())

		self.model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
		self.model.add(GaussianNoise(0.1))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())

		self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.1))

		self.model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.2))

		self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.2))

		self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.2))

		self.model.add(Flatten())
		self.model.add(Dense(512))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(Dropout(0.5))
		self.model.add(Dense(256))
		self.model.add(LeakyReLU())
		self.model.add(Dense(7))
		self.model.add(Activation('softmax'))
		adam = Adam(lr=self.lr, decay=self.decay)
		self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


	def train(self, trainx, trainy):
		self.__graph__()

		# augmentation
		datagen = ImageDataGenerator(
			rotation_range=30.0,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.1,
			zoom_range=0.2,
			horizontal_flip=True,
			fill_mode='constant',
			vertical_flip=False)

		datagen.fit(trainx)

		datagen_flow = datagen.flow(x=trainx, y=trainy, batch_size=128)

		for i in range(self.epoch // 100):
			self.model.fit_generator(datagen_flow,steps_per_epoch=trainx.shape[0] / 128,
					initial_epoch=i * 100,epochs=(i + 1) * 100)


	def predict(self, testx):
		return np.argmax(self.model.predict(testx), axis=1).reshape(-1, 1)

class model2:
	
	def __init__(self, lr, decay, epoch):
		self.lr = lr
		self.decay = decay
		self.epoch = epoch

	def __graph__(self):
		self.model = Sequential()
		self.model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(48,48,1), kernel_initializer='he_normal'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())

		self.model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())

		self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))

		self.model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))

		self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))

		self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))

		self.model.add(Flatten())
		self.model.add(Dense(512))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(Dropout(0.5))
		self.model.add(Dense(256))
		self.model.add(LeakyReLU())
		self.model.add(Dense(7))
		self.model.add(Activation('softmax'))
		adam = Adam(lr=self.lr, decay=self.decay)
		self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


	def train(self, trainx, trainy):
		self.__graph__()

		# augmentation
		datagen = ImageDataGenerator(
			rotation_range=20.0,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.1,
			zoom_range=0.2,
			horizontal_flip=True,
			fill_mode='constant',
			vertical_flip=False)

		datagen.fit(trainx)

		datagen_flow = datagen.flow(x=trainx, y=trainy, batch_size=128)

		cycle = 70
		for i in range(self.epoch // 70):
			self.model.fit_generator(datagen_flow,steps_per_epoch=trainx.shape[0] / 128,
					initial_epoch=i * cycle,epochs=(i + 1) * cycle)


	def predict(self, testx):
		return np.argmax(self.model.predict(testx), axis=1).reshape(-1, 1)

class model3:
	
	def __init__(self, lr, decay, epoch):
		self.lr = lr
		self.decay = decay
		self.epoch = epoch

	def __graph__(self):
		self.model = Sequential()
		self.model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(48,48,1), kernel_initializer='he_normal'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())

		self.model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'))
		self.model.add(GaussianNoise(0.1))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())

		self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))

		self.model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))

		self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))

		self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'))
		self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))

		self.model.add(Flatten())
		self.model.add(Dense(512))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(Dropout(0.5))
		self.model.add(Dense(512))
		self.model.add(LeakyReLU())
		self.model.add(Dense(7))
		self.model.add(Activation('softmax'))
		adam = Adam(lr=self.lr, decay=self.decay)
		self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


	def train(self, trainx, trainy):
		self.__graph__()

		# augmentation
		datagen = ImageDataGenerator(
			rotation_range=20.0,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.1,
			zoom_range=0.2,
			horizontal_flip=True,
			fill_mode='constant',
			vertical_flip=False)

		datagen.fit(trainx)

		datagen_flow = datagen.flow(x=trainx, y=trainy, batch_size=128)

		cycle = 70
		for i in range(self.epoch // 70):
			self.model.fit_generator(datagen_flow,steps_per_epoch=trainx.shape[0] / 128,
					initial_epoch=i * cycle,epochs=(i + 1) * cycle)


	def predict(self, testx):
		return np.argmax(self.model.predict(testx), axis=1).reshape(-1, 1)

class model4:
	
	def __init__(self, lr, decay, epoch):
		self.lr = lr
		self.decay = decay
		self.epoch = epoch

	def __graph__(self):
		self.model = Sequential()
		self.model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(48,48,1), activation='selu'))
		self.model.add(GaussianNoise(0.1))
		self.model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='selu'))
		self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='selu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.1))
		self.model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='selu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.2))
		self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='selu'))
		self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='selu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.2))

		self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='selu'))
		self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='selu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.2))

		self.model.add(Flatten())
		self.model.add(Dense(512))
		self.model.add(BatchNormalization(axis=-1, momentum=0.5))
		self.model.add(LeakyReLU())
		self.model.add(Dropout(0.5))
		self.model.add(Dense(256))
		self.model.add(LeakyReLU())
		self.model.add(Dense(7))
		self.model.add(Activation('softmax'))
		adam = Adam(lr=self.lr, decay=self.decay)
		self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


	def train(self, trainx, trainy):
		self.__graph__()

		# augmentation
		datagen = ImageDataGenerator(
			rotation_range=30.0,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.1,
			zoom_range=0.2,
			horizontal_flip=True,
			fill_mode='constant',
			vertical_flip=False)

		datagen.fit(trainx)

		datagen_flow = datagen.flow(x=trainx, y=trainy, batch_size=128)

		for i in range(self.epoch // 50):
			self.model.fit_generator(datagen_flow,steps_per_epoch=trainx.shape[0] / 128,
					initial_epoch=i * 50,epochs=(i + 1) * 50)

	def predict(self, testx):
		return np.argmax(self.model.predict(testx), axis=1).reshape(-1, 1)

		
