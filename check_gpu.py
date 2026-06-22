import torch

print(f"PyTorch verzió: {torch.__version__}")

# Ellenőrizzük, hogy a CUDA elérhető-e a PyTorch számára
is_available = torch.cuda.is_available()
print(f"CUDA elérhető a PyTorch számára? -> {is_available}")

if is_available:
    # Hány GPU-t lát a rendszer?
    device_count = torch.cuda.device_count()
    print(f"Észlelt GPU-k száma: {device_count}")
    
    # Melyik az aktuális GPU?
    current_device = torch.cuda.current_device()
    print(f"Aktuális eszköz indexe: {current_device}")
    
    # Mi a neve a GPU-nak?
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Aktuális eszköz neve: {device_name}")
else:
    print("A PyTorch nem talál CUDA-képes eszközt. A műveletek CPU-n fognak futni.")
