from Basic_Demo.Data_of_MNIST_in_Analysis import run

image_data = run()

x_train = image_data.train_images_1
y_train = image_data.train_labels_1
x_valid = image_data.test_images_1
y_valid = image_data.test_labels_1

