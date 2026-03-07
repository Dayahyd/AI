import tensorflow as tf
import numpy as np

# Create two 3x3 matrices using tf.constant
matrix_a = tf.constant([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], dtype=tf.float32)

matrix_b = tf.constant([[9, 8, 7],
                        [6, 5, 4],
                        [3, 2, 1]], dtype=tf.float32)

# Compute the sum of the two matrices
sum_matrix = tf.add(matrix_a, matrix_b)

# Calculate the determinant of the resulting matrix
determinant = tf.linalg.det(sum_matrix)

# Print the results
print("Matrix A:")
print(matrix_a.numpy())
print("\nMatrix B:")
print(matrix_b.numpy())
print("\nSum Matrix (A + B):")
print(sum_matrix.numpy())
print("\nDeterminant of the sum matrix:")
print(determinant.numpy())

