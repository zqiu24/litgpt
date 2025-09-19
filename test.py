import torch

# Create another sample 3D tensor
x = torch.zeros(2, 3, 3)
print("Original 3D tensor:\n", x)
print(f"Memory address of x: {x.data_ptr()}")

# Get a view of the diagonals for each matrix in the batch
# and add 1 to them in-place.
torch.diagonal(x, dim1=-2, dim2=-1).add_(1)

print("\nResult after in-place diagonal modification:\n", x)
print(f"Memory address of x: {x.data_ptr()}")