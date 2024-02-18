
import torch
print(torch.cuda.is_available())  # Check if CUDA is available
print(torch.cuda.device_count())  # Print the number of GPUs available
print(torch.cuda.get_device_name(0))  # Print the name of the first GPU
