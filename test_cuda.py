import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should be >0
print(torch.cuda.current_device())  # Should print device ID
print(torch.cuda.get_device_name(0))  # Should print GPU name
