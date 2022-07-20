# import cv2
# import numpy as np
# from numpy import ndarray
# from tensorflow import keras


# def cell_is_empty(cell: ndarray):
#     cell: ndarray = cell[15:35, 15:35]
#     return len(cell[cell == 1]) == 0


# def classify_cells(sudoku_fields):
#     model = keras.models.load_model('keras-models/mnist_model0')

#     numbers = []
#     for field in sudoku_fields:
#         if cell_is_empty(field):
#             numbers.append(0)
#         else:
#             image = cv2.resize(field, (28, 28))
#             image = image.astype('float32')
#             image = image.reshape(1, 28, 28, 1)
#             y_prob = model.predict(image)
#             y_class = y_prob.argmax(axis=1)
#             numbers.append(y_class[0])

#     numbers = np.array(numbers)
#     numbers = np.transpose(numbers.reshape((9, 9)))
#     return numbers
