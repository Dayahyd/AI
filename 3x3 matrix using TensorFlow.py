import tensorflow as tf

# Define the 3x3 matrix values
matrix_values = [[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]

# Create a TensorFlow constant tensor from the values
matrix_constant = tf.constant(matrix_values, dtype=tf.float32) 

print("Fixed 3x3 Matrix (tf.constant):")
print(matrix_constant)
