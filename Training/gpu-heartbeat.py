import torch

def check_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        num_processors = torch.cuda.get_device_properties(0).multi_processor_count
        print("🚀 GPU found!")
        print(f"Device: GPU ({device_name})")
        print(f"Number of processors: {num_processors}")
    else:
        print("😢 No GPU found...")
        print("🐌 Looks like we're CPU-crawling today.")

if __name__ == "__main__":
    check_device()