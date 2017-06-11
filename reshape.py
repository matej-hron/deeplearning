from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28)) 
train_images = train_images.astype('float32') / 255

print(train_images)
