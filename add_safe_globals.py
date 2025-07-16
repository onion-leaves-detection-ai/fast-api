import torch
print(torch.__version__)  # ✅ Should print 2.1.0
print(hasattr(torch.serialization, "add_safe_globals"))  # ✅ Should print True
