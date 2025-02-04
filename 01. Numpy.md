NumPy Cheatsheet with essential functions, examples, and common scenarios.
________________________________________
📌 NumPy Cheatsheet 🧮
NumPy (Numerical Python) is a library for numerical computing in Python.
1️⃣ Import NumPy
import numpy as np
________________________________________
🛠 Array Creation
# 1D Array
arr1 = np.array([1, 2, 3])

# 2D Array (Matrix)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# 3D Array
arr3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Zeros, Ones, Empty, Identity Matrix
zeros = np.zeros((2, 3))     # 2x3 matrix of zeros
ones = np.ones((3, 3))       # 3x3 matrix of ones
empty = np.empty((2, 2))     # Uninitialized array
identity = np.eye(4)         # 4x4 identity matrix

# Range & Linspace
arange_arr = np.arange(1, 10, 2)  # [1, 3, 5, 7, 9]
linspace_arr = np.linspace(0, 1, 5)  # [0. 0.25 0.5 0.75 1.]
________________________________________
📏 Array Shape & Reshape
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)      # (2, 3) → 2 rows, 3 columns
print(arr.ndim)       # 2 → Number of dimensions
print(arr.size)       # 6 → Total elements

# Reshaping
reshaped = arr.reshape((3, 2))  # Change shape to 3x2
flattened = arr.ravel()         # Flatten array into 1D
________________________________________
🔢 Data Types
arr = np.array([1, 2, 3], dtype=np.float64)  # Set type explicitly
print(arr.dtype)  # float64

arr = arr.astype(np.int32)  # Convert type
print(arr.dtype)  # int32
________________________________________
🎯 Indexing & Slicing
arr = np.array([10, 20, 30, 40, 50])

print(arr[1])      # 20 (Indexing)
print(arr[-1])     # 50 (Last element)
print(arr[1:4])    # [20, 30, 40] (Slicing)
print(arr[::-1])   # [50, 40, 30, 20, 10] (Reverse)
arr2D = np.array([[1, 2, 3], [4, 5, 6]])

print(arr2D[1, 2])     # 6 → Row index 1, Column index 2
print(arr2D[:, 1])     # [2, 5] → All rows, column 1
print(arr2D[0, :])     # [1, 2, 3] → First row
________________________________________
🔄 Operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise Operations
print(a + b)  # [5, 7, 9]
print(a - b)  # [-3, -3, -3]
print(a * b)  # [4, 10, 18]
print(a / b)  # [0.25, 0.4, 0.5]

# Scalar Operations
print(a * 2)  # [2, 4, 6]

# Matrix Multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))   # [[19 22] [43 50]]

# Broadcasting
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr + np.array([10, 20, 30]))  
# [[11 22 33] 
#  [14 25 36]]
________________________________________
📊 Statistical Functions
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(np.sum(arr))    # 21
print(np.mean(arr))   # 3.5
print(np.median(arr)) # 3.5
print(np.std(arr))    # Standard deviation
print(np.var(arr))    # Variance
print(np.min(arr))    # 1
print(np.max(arr))    # 6

# Row-wise and Column-wise Operations
print(np.sum(arr, axis=0))  # Column-wise sum [5, 7, 9]
print(np.sum(arr, axis=1))  # Row-wise sum [6, 15]
________________________________________
🔍 Boolean & Conditional Operations
arr = np.array([1, 2, 3, 4, 5, 6])

# Boolean Masking
mask = arr > 3
print(arr[mask])   # [4, 5, 6]

# Where condition
print(np.where(arr > 3, arr * 2, arr / 2))  
# [ 0.5  1.   1.5  8.  10.  12. ]
________________________________________
🏗 Stacking & Splitting
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Stacking Arrays
print(np.vstack((a, b)))  # Vertical stack
print(np.hstack((a, b)))  # Horizontal stack

# Splitting Arrays
arr = np.array([1, 2, 3, 4, 5, 6])
print(np.split(arr, 3))  # [array([1, 2]), array([3, 4]), array([5, 6])]
________________________________________
🔢 Sorting & Unique
arr = np.array([5, 2, 8, 1, 9])

print(np.sort(arr))   # [1, 2, 5, 8, 9]
print(np.argsort(arr)) # Indices of sorted order

arr2 = np.array([[3, 2, 1], [6, 5, 4]])
print(np.sort(arr2, axis=1))  # Sort along rows
print(np.sort(arr2, axis=0))  # Sort along columns

# Unique values
arr_dup = np.array([1, 2, 2, 3, 4, 4, 5])
print(np.unique(arr_dup))  # [1, 2, 3, 4, 5]
________________________________________
🏁 Random Number Generation
np.random.seed(42)  # Set seed for reproducibility

print(np.random.rand(3, 3))  # Uniform [0,1]
print(np.random.randn(3, 3))  # Normal distribution
print(np.random.randint(1, 10, (2, 2)))  # Integers in range

arr = np.array([1, 2, 3, 4, 5])
print(np.random.choice(arr))  # Random element from array
________________________________________
🔄 Saving & Loading Data
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Save & Load as .npy
np.save("array.npy", arr)
loaded_arr = np.load("array.npy")

# Save & Load as .csv
np.savetxt("array.csv", arr, delimiter=",")
loaded_csv = np.loadtxt("array.csv", delimiter=",")
________________________________________
💡 Use Cases
✔️ Data Analysis
✔️ Machine Learning
✔️ Image Processing
✔️ Finance Calculations
________________________________________
🔹 This cheatsheet covers all essential NumPy operations with examples. 🚀

