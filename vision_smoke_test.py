import sys
import torch


def main() -> int:
    try:
        import open_clip  # noqa: F401
    except Exception as exc:
        print(f"SKIP: open_clip_torch not available ({exc}).")
        return 0

    from vision_encoder import TimeVLMVisionEncoder

    encoder = TimeVLMVisionEncoder(
        input_dim=7,
        image_size=224,
        periodicity=24,
        hidden_dim=64,
        output_channels=3,
        clip_model_name="ViT-B-32",
        clip_pretrained="openai",
        freeze_clip=True,
    )

    x_enc = torch.randn(2, 32, 7)
    with torch.no_grad():
        out = encoder(x_enc)
    print("vision_out", tuple(out.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
