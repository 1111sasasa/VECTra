import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from timeseries_image import LearnableTimeSeriesToImage


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class TimeVLMVisionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        image_size: int,
        periodicity: int,
        hidden_dim: int,
        output_channels: int,
        clip_model_name: str,
        clip_pretrained: str,
        freeze_clip: bool = False,
        freeze_ts_to_image: bool = False,
    ) -> None:
        super().__init__()

        try:
            import open_clip
        except Exception as exc:
            raise ImportError(
                "open_clip_torch is required for the vision encoder."
            ) from exc

        self.ts_to_image = LearnableTimeSeriesToImage(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_channels=output_channels,
            image_size=image_size,
            periodicity=periodicity,
        )
        self.freeze_ts_to_image = freeze_ts_to_image
        if self.freeze_ts_to_image:
            for param in self.ts_to_image.parameters():
                param.requires_grad = False
            self.ts_to_image.eval()

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.freeze_clip = freeze_clip
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()

        output_dim = getattr(self.clip_model.visual, "output_dim", None)
        if output_dim is None:
            output_dim = getattr(self.clip_model, "embed_dim", None)
        if output_dim is None:
            raise ValueError("Unable to infer CLIP output dimension.")
        self.output_dim = int(output_dim)

        mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean)
        self.register_buffer("clip_std", std)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        if self.freeze_ts_to_image:
            self.ts_to_image.eval()
            with torch.no_grad():
                images = self.ts_to_image(x_enc)
        else:
            images = self.ts_to_image(x_enc)
        if images.shape[1] != 3:
            raise ValueError("Vision encoder expects 3-channel images.")

        # Remove per-image min-max normalization to preserve amplitude information
        # img_min = images.amin(dim=(2, 3), keepdim=True)
        # img_max = images.amax(dim=(2, 3), keepdim=True)
        # images = (images - img_min) / (img_max - img_min + 1e-6)

        # normalize with CLIP mean and std
        images = (images - self.clip_mean) / self.clip_std

        if self.freeze_clip:
            self.clip_model.eval()
            with torch.no_grad():
                return self.clip_model.encode_image(images)

        return self.clip_model.encode_image(images)


class SegmentVisionEncoder(nn.Module):
    def __init__(self, base_encoder: TimeVLMVisionEncoder, hidden_dim: int):
        super().__init__()
        self.base_encoder = base_encoder
        self.output_dim = base_encoder.output_dim
        # Projection is handled in JGRM model usually, but can be here too.

    def forward(self, window_tensors, batch_route_lengths):
        """
        window_tensors: List of tensors (variable length GPS sequences).
        batch_route_lengths: List of ints (segments per traj).

        Returns:
            segment_embs: (Batch, MaxSegLen, Hidden)
            traj_emb: (Batch, Hidden) - Pooled from segments or CLS
        """
        # Batch the windows
        # Pad sequence
        padded_windows = rnn_utils.pad_sequence(
            window_tensors, batch_first=True, padding_value=0.0
        )
        # (TotalSegs, MaxTime, Feats)

        # Encode
        # TimeVLMVisionEncoder expects (B, T, D) -> (B, Embed)
        seg_embeddings = self.base_encoder(padded_windows)  # (TotalSegs, Embed)

        # Reconstruct into (Batch, MaxSegLen, Embed)
        max_segs = max(batch_route_lengths)
        batch_size = len(batch_route_lengths)

        output_tensor = torch.zeros(
            (batch_size, max_segs, seg_embeddings.size(-1)),
            device=seg_embeddings.device,
            dtype=seg_embeddings.dtype,
        )

        start = 0
        traj_embs_list = []

        for i, length in enumerate(batch_route_lengths):
            if length == 0:
                traj_embs_list.append(
                    torch.zeros(seg_embeddings.size(-1), device=seg_embeddings.device)
                )
                continue

            end = start + length
            segs = seg_embeddings[start:end]
            output_tensor[i, :length, :] = segs

            # Trajectory level pooling (Mean)
            traj_mean = segs.mean(dim=0)
            traj_embs_list.append(traj_mean)

            start = end

        traj_emb = torch.stack(traj_embs_list, dim=0)  # (Batch, Embed)

        return output_tensor, traj_emb

