from transformers import AutoProcessor
import torch


MODEL_ID = "google/gemma-4-E2B-it"


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device={torch.cuda.get_device_name(0)}")
        print(f"vram_gb={torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"processor={processor.__class__.__name__}")


if __name__ == "__main__":
    main()

