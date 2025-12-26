import torch
import time

def measure_single_image_latency(model, input_shape=(1,1,128,128), device="cuda", repeat=100):
    model = model.to(device)
    model.eval()

    dummy = torch.randn(*input_shape).to(device)

    # GPU warm-up
    for _ in range(10):
        _ = model(dummy)
    torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(dummy)
        torch.cuda.synchronize()
    end = time.time()

    latency = (end - start) / repeat
    print(f"Single Image Latency: {latency*1000:.3f} ms")
    return latency
