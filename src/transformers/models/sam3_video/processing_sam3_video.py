# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union

import torch
from torchvision.ops import masks_to_boxes

from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires
from ...video_utils import VideoInput
from .modeling_sam3_video import Sam3VideoInferenceSession


@requires(backends=("torch",))
@auto_docstring
class Sam3VideoProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor,
        video_processor,
        tokenizer,
        target_size: int | None = None,
        **kwargs,
    ):
        r"""
        target_size (`int`, *optional*):
            The target size (target_size, target_size) to which the image will be resized.
        """
        super().__init__(image_processor, video_processor, tokenizer, **kwargs)
        self.target_size = target_size if target_size is not None else self.image_processor.size["height"]

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        segmentation_maps: ImageInput | None = None,
        original_sizes: list[list[float]] | torch.Tensor | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        images (`ImageInput`, *optional*):
            The image(s) to process.
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to process (optional, for image processor).
        original_sizes (`list[list[float]]`, `torch.Tensor`, *optional*):
            The original sizes of the images. Only used when images is not provided.

        Returns:
            A [`BatchEncoding`] with the following fields:
            - `pixel_values` (`torch.Tensor`): The processed image(s).
            - `original_sizes` (`list[list[float]]`): The original sizes of the images.
            - `labels` (`torch.Tensor`, *optional*): The processed segmentation maps (if provided).
        """
        if images is not None:
            encoding_image_processor = self.image_processor(
                images,
                segmentation_maps=segmentation_maps,
                return_tensors=return_tensors,
                **kwargs,
            )
        elif original_sizes is not None:
            if isinstance(original_sizes, torch.Tensor):
                original_sizes = original_sizes.cpu().tolist()
            encoding_image_processor = BatchEncoding({"original_sizes": original_sizes}, tensor_type=return_tensors)
        else:
            raise ValueError("Either images or original_sizes must be provided")

        original_sizes = encoding_image_processor["original_sizes"]
        # Check original_sizes is of length 1 or len(images)
        if images is not None and len(original_sizes) != 1 and len(original_sizes) != len(images):
            raise ValueError(
                "original_sizes must be of length 1 or len(images). If you are passing a single image, you must pass a single original_size."
            )

        return encoding_image_processor

    def _resolve_original_size_for_prompt(
        self,
        inference_session: Sam3VideoInferenceSession,
        original_size: tuple[int, int] | list[int] | torch.Tensor | None,
    ) -> tuple[int, int]:
        if original_size is None:
            if inference_session.video_height is None or inference_session.video_width is None:
                raise ValueError(
                    "Either inference_session.video_height/video_width must be set, or original_size must be provided."
                )
            return int(inference_session.video_height), int(inference_session.video_width)

        if isinstance(original_size, torch.Tensor):
            original_size = original_size.cpu().tolist()

        if isinstance(original_size, (list, tuple)):
            if len(original_size) == 2 and all(isinstance(v, (int, float)) for v in original_size):
                return int(original_size[0]), int(original_size[1])
            if (
                len(original_size) == 1
                and isinstance(original_size[0], (list, tuple))
                and len(original_size[0]) == 2
                and all(isinstance(v, (int, float)) for v in original_size[0])
            ):
                return int(original_size[0][0]), int(original_size[0][1])

        raise ValueError(f"original_size must be [height, width] (or [[height, width]]), got {original_size!r}.")

    def _prepare_prompt_boxes(
        self,
        input_boxes: list[list[float]] | torch.Tensor,
        input_boxes_labels: list[int] | torch.Tensor | None,
        original_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(input_boxes, torch.Tensor):
            boxes = input_boxes.detach().clone().to(dtype=torch.float32)
        else:
            boxes = torch.tensor(input_boxes, dtype=torch.float32)

        if boxes.ndim == 1:
            if boxes.numel() != 4:
                raise ValueError(f"input_boxes must have 4 coordinates per box, got shape {tuple(boxes.shape)}.")
            boxes = boxes.unsqueeze(0)
        if boxes.ndim == 3 and boxes.shape[0] == 1:
            boxes = boxes.squeeze(0)
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(
                "input_boxes must have shape [num_boxes, 4] (or [4] for a single box), "
                f"got shape {tuple(boxes.shape)}."
            )

        old_h, old_w = original_size

        # Validate box coordinates before normalization
        x0, y0, x1, y1 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        invalid_boxes = (x1 < x0) | (y1 < y0)
        if torch.any(invalid_boxes):
            invalid_idx = torch.where(invalid_boxes)[0].tolist()
            raise ValueError(
                f"input_boxes must be in valid XYXY format with x1 >= x0 and y1 >= y0. "
                f"Found invalid boxes at indices: {invalid_idx}"
            )

        # Normalize coordinates
        boxes[..., 0] = boxes[..., 0] / old_w
        boxes[..., 1] = boxes[..., 1] / old_h
        boxes[..., 2] = boxes[..., 2] / old_w
        boxes[..., 3] = boxes[..., 3] / old_h
        x0, y0, x1, y1 = boxes.unbind(-1)
        boxes = torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)), dim=-1).unsqueeze(0)

        box_labels = None
        if input_boxes_labels is not None:
            if isinstance(input_boxes_labels, torch.Tensor):
                box_labels = input_boxes_labels.detach().clone().to(dtype=torch.int64)
            else:
                box_labels = torch.tensor(input_boxes_labels, dtype=torch.int64)

            if box_labels.ndim == 0:
                box_labels = box_labels.unsqueeze(0)
            if box_labels.ndim == 2 and box_labels.shape[0] == 1:
                box_labels = box_labels.squeeze(0)
            if box_labels.ndim != 1:
                raise ValueError(
                    "input_boxes_labels must have shape [num_boxes] (or scalar for one box), "
                    f"got shape {tuple(box_labels.shape)}."
                )
            if box_labels.shape[0] != boxes.shape[1]:
                raise ValueError(
                    f"Number of box labels ({box_labels.shape[0]}) must match number of boxes ({boxes.shape[1]})."
                )
            box_labels = box_labels.unsqueeze(0)

        return boxes, box_labels

    def add_prompt(
        self,
        inference_session: Sam3VideoInferenceSession,
        text: str | None = None,
        input_boxes: list[list[float]] | torch.Tensor | None = None,
        input_boxes_labels: list[int] | torch.Tensor | None = None,
        original_size: tuple[int, int] | list[int] | torch.Tensor | None = None,
        prompt_id: int | None = None,
        deduplicate_text: bool = False,
    ) -> int:
        """
        Add or update a prompt in the inference session.

        Args:
            inference_session (`Sam3VideoInferenceSession`):
                The inference session.
            text (`str`, *optional*):
                Text prompt for detection. If omitted while `input_boxes` are provided, an existing prompt text
                is reused when `prompt_id` is provided, otherwise `"visual"` is used.
            input_boxes (`list[list[float]]` or `torch.Tensor`, *optional*):
                Bounding box prompt(s) in absolute XYXY format for one prompt.
                If provided for an existing `prompt_id`, replaces any previously stored boxes.
            input_boxes_labels (`list[int]` or `torch.Tensor`, *optional*):
                Optional labels for the input boxes.
            original_size (`tuple[int, int]`, `list[int]`, or `torch.Tensor`, *optional*):
                Original frame size as `(height, width)`. Required if the inference session has no video size.
            prompt_id (`int`, *optional*):
                Explicit prompt ID to update. If omitted, a new prompt ID is created unless deduplicated.
            deduplicate_text (`bool`, *optional*, defaults to `False`):
                Whether to reuse an existing prompt ID for identical text (exact match).

        Returns:
            `int`: Prompt ID.
        """
        if text is None and input_boxes is None:
            raise ValueError("At least one of text or input_boxes must be provided.")
        if input_boxes is None and input_boxes_labels is not None:
            raise ValueError("input_boxes_labels cannot be provided when input_boxes is None.")
        if text is not None and not isinstance(text, str):
            raise ValueError(f"text must be a string when provided, got {type(text)}.")
        if prompt_id is not None and prompt_id < 0:
            raise ValueError(f"prompt_id must be >= 0, got {prompt_id}.")

        if text is not None:
            prompt_text = text
        elif prompt_id is not None and prompt_id in inference_session.prompts:
            prompt_text = inference_session.prompts[prompt_id]
        else:
            prompt_text = "visual"
        boxes = None
        boxes_labels = None
        if input_boxes is not None:
            resolved_original_size = self._resolve_original_size_for_prompt(inference_session, original_size)
            boxes, boxes_labels = self._prepare_prompt_boxes(input_boxes, input_boxes_labels, resolved_original_size)

        prompt_id = inference_session.add_prompt(
            prompt_text=prompt_text,
            prompt_id=prompt_id,
            deduplicate_text=deduplicate_text,
        )

        if prompt_id not in inference_session.prompt_input_ids:
            encoded_text = self.tokenizer(prompt_text, return_tensors="pt", padding="max_length", max_length=32).to(
                inference_session.inference_device
            )
            inference_session.prompt_input_ids[prompt_id] = encoded_text.input_ids
            inference_session.prompt_attention_masks[prompt_id] = encoded_text.attention_mask

        if boxes is not None:
            inference_session.prompt_input_boxes[prompt_id] = boxes.to(inference_session.inference_device)
            if boxes_labels is not None:
                inference_session.prompt_input_boxes_labels[prompt_id] = boxes_labels.to(
                    inference_session.inference_device
                )
            else:
                inference_session.prompt_input_boxes_labels.pop(prompt_id, None)

        return prompt_id

    def add_text_prompt(self, inference_session: Sam3VideoInferenceSession, text: str | list[str]):
        """
        Add text prompt(s) to the inference session.

        Args:
            inference_session (`Sam3VideoInferenceSession`): The inference session.
            text (`str` or `list[str]`): The text prompt(s) to add.

        Returns:
            `Sam3VideoInferenceSession`: The inference session with the added text prompt(s).
        """
        if isinstance(text, str):
            text = [text]

        for prompt_text in text:
            self.add_prompt(
                inference_session=inference_session,
                text=prompt_text,
                deduplicate_text=True,
            )

        return inference_session

    def add_box_prompt(
        self,
        inference_session: Sam3VideoInferenceSession,
        input_boxes: list[list[float]] | torch.Tensor,
        input_boxes_labels: list[int] | torch.Tensor | None = None,
        original_size: tuple[int, int] | list[int] | torch.Tensor | None = None,
        prompt_id: int | None = None,
    ):
        """
        Add or update a box prompt in the inference session.

        Args:
            inference_session (`Sam3VideoInferenceSession`):
                The inference session.
            input_boxes (`list[list[float]]` or `torch.Tensor`):
                Bounding box prompt(s) in absolute XYXY format for one prompt.
            input_boxes_labels (`list[int]` or `torch.Tensor`, *optional*):
                Optional labels for input boxes.
            original_size (`tuple[int, int]`, `list[int]`, or `torch.Tensor`, *optional*):
                Original frame size as `(height, width)`. Required if the inference session has no video size.
            prompt_id (`int`, *optional*):
                Existing prompt ID to update. When omitted, a new prompt is created with text
                `"visual"` (the Sam3 convention for box-only prompts).

        Returns:
            `Sam3VideoInferenceSession`: The inference session with the added/updated box prompt.
        """
        self.add_prompt(
            inference_session=inference_session,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            original_size=original_size,
            prompt_id=prompt_id,
        )
        return inference_session

    def init_video_session(
        self,
        video: VideoInput | None = None,
        inference_device: Union[str, "torch.device"] = "cpu",
        inference_state_device: Union[str, "torch.device"] | None = None,
        processing_device: Union[str, "torch.device"] | None = None,
        video_storage_device: Union[str, "torch.device"] | None = None,
        max_vision_features_cache_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes a video session for inference.
        If a video is provided (async inference), the video will be processed and stored on the `video_storage_device`.

        Args:
            video (`VideoInput`, *optional*):
                The video to process. No need to provide when streaming.
            inference_device (`str` or `torch.device`, *optional*, defaults to "cpu"):
                The device to use for inference.
            inference_state_device (`str` or `torch.device`, *optional*):
                The device to store the inference state on.
            processing_device (`str` or `torch.device`, *optional*):
                The device to use for video processing.
            video_storage_device (`str` or `torch.device`, *optional*):
                The device to store the processed video frames on.
            max_vision_features_cache_size (`int`, *optional*, defaults to 1):
                The maximum number of vision features to cache.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                The torch dtype to use for the whole session.
        """
        video_storage_device = video_storage_device if video_storage_device is not None else inference_device
        inference_state_device = inference_state_device if inference_state_device is not None else inference_device
        processing_device = processing_device if processing_device is not None else inference_device
        pixel_values_video = None
        video_height = None
        video_width = None
        if video is not None:
            processed_video = self.video_processor(videos=video, device=processing_device, return_tensors="pt")
            pixel_values_video = processed_video.pixel_values_videos[0]
            video_height = processed_video.original_sizes[0][0]
            video_width = processed_video.original_sizes[0][1]
        inference_session = Sam3VideoInferenceSession(
            video=pixel_values_video,
            video_height=video_height,
            video_width=video_width,
            inference_device=inference_device,
            video_storage_device=video_storage_device,
            inference_state_device=inference_state_device,
            dtype=dtype,
            max_vision_features_cache_size=max_vision_features_cache_size,
        )
        return inference_session

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def _apply_object_wise_non_overlapping_constraints(
        self,
        pred_masks,
        obj_scores,
        background_value=-10.0,
        prompt_ids=None,
    ):
        """
        Applies non-overlapping constraints object wise (i.e. only one object can claim the overlapping region).
        Constraints are enforced independently for each prompt group when `prompt_ids` are provided.
        """
        if prompt_ids is None:
            return self._apply_object_wise_non_overlapping_constraints_impl(pred_masks, obj_scores, background_value)

        if len(prompt_ids) != pred_masks.size(0):
            raise ValueError("prompt_ids must have the same length as pred_masks")

        pred_masks_grouped = pred_masks.clone()
        prompt_ids_tensor = torch.tensor(prompt_ids, device=pred_masks.device, dtype=torch.long)
        for prompt_id in prompt_ids_tensor.unique(sorted=True):
            indices = torch.nonzero(prompt_ids_tensor == prompt_id, as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            prompt_masks = self._apply_object_wise_non_overlapping_constraints_impl(
                pred_masks_grouped[indices],
                obj_scores[indices],
                background_value,
            )
            pred_masks_grouped[indices] = prompt_masks.to(pred_masks_grouped.dtype)
        return pred_masks_grouped

    def _apply_object_wise_non_overlapping_constraints_impl(self, pred_masks, obj_scores, background_value=-10.0):
        pred_masks_single_score = torch.where(pred_masks > 0, obj_scores[..., None, None], background_value)
        pixel_level_non_overlapping_masks = self._apply_non_overlapping_constraints(pred_masks_single_score)
        pred_masks = torch.where(
            pixel_level_non_overlapping_masks > 0,
            pred_masks,
            torch.clamp(pred_masks, max=background_value),
        )
        return pred_masks.to(pred_masks_single_score.dtype)

    def postprocess_outputs(
        self,
        inference_session,
        model_outputs,
        original_sizes: list[list[float]] | torch.Tensor | None = None,
    ):
        """
        Post-process model outputs to get final masks, boxes, and scores.

        Args:
            inference_session (`Sam3VideoInferenceSession`):
                The inference session object.
            model_outputs (`Sam3VideoSegmentationOutput`):
                The raw model output from `Sam3VideoModel.forward()`.
            original_sizes (`list[list[float]]` or `torch.Tensor`, *optional*):
                Optional original frame sizes [height, width]. Required for streaming inference
                when video_height/video_width are not set in the session.

        Returns:
            `dict`: A dictionary containing the following keys:
                - **object_ids** (`torch.Tensor` of shape `(num_objects,)`): Object IDs for each detected object.
                - **scores** (`torch.Tensor` of shape `(num_objects,)`): Detection scores for each object.
                - **boxes** (`torch.Tensor` of shape `(num_objects, 4)`): Bounding boxes in XYXY format
                  (top_left_x, top_left_y, bottom_right_x, bottom_right_y).
                - **masks** (`torch.Tensor` of shape `(num_objects, height, width)`): Binary segmentation masks
                  for each object at the original video resolution.
                - **prompt_id_to_obj_ids** (`dict[int, list[int]]`): Mapping from prompt ID to list of
                  object IDs detected by that prompt.
                - **prompt_to_obj_ids** (`dict[str, list[int]]`): Compatibility mapping from prompt text to list of
                  object IDs detected by prompts with that text.
        """
        obj_id_to_mask = model_outputs["obj_id_to_mask"]  # low res masks (1, H_low, W_low)
        curr_obj_ids = sorted(obj_id_to_mask.keys())

        # Get video dimensions - use original_sizes for streaming inference if session doesn't have them
        if inference_session.video_height is not None and inference_session.video_width is not None:
            H_video, W_video = inference_session.video_height, inference_session.video_width
        elif original_sizes is not None:
            if isinstance(original_sizes, torch.Tensor):
                original_sizes = original_sizes.cpu().tolist()
            # original_sizes is a list of [height, width] pairs, take the first one
            if isinstance(original_sizes[0], list):
                H_video, W_video = int(original_sizes[0][0]), int(original_sizes[0][1])
            else:
                H_video, W_video = int(original_sizes[0]), int(original_sizes[1])
        else:
            raise ValueError(
                "Either inference_session.video_height/video_width must be set, "
                "or original_sizes must be provided for streaming inference."
            )
        if len(curr_obj_ids) == 0:
            out_obj_ids = torch.zeros(0, dtype=torch.int64)
            out_probs = torch.zeros(0, dtype=torch.float32)
            out_binary_masks = torch.zeros(0, H_video, W_video, dtype=torch.bool)
            out_boxes_xyxy = torch.zeros(0, 4, dtype=torch.float32)
        else:
            out_obj_ids = torch.tensor(curr_obj_ids, dtype=torch.int64)
            out_probs = torch.tensor([model_outputs["obj_id_to_score"][obj_id] for obj_id in curr_obj_ids])
            out_tracker_probs = torch.tensor(
                [model_outputs["obj_id_to_tracker_score"].get(obj_id, 0.0) for obj_id in curr_obj_ids]
            )

            # Interpolate low-res masks to video resolution
            low_res_masks = torch.cat([obj_id_to_mask[obj_id] for obj_id in curr_obj_ids], dim=0)  # (N, H_low, W_low)
            # Add channel dimension for interpolation: (N, H, W) -> (N, 1, H, W)
            out_binary_masks = torch.nn.functional.interpolate(
                low_res_masks.unsqueeze(1),
                size=(H_video, W_video),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # (N, H_video, W_video)
            out_binary_masks = out_binary_masks > 0

            assert out_binary_masks.dtype == torch.bool
            keep = out_binary_masks.any(dim=(1, 2)).cpu()  # remove masks with 0 areas
            # hide outputs for those object IDs in `obj_ids_to_hide`
            obj_ids_to_hide = []
            if model_outputs["suppressed_obj_ids"] is not None:
                obj_ids_to_hide.extend(list(model_outputs["suppressed_obj_ids"]))
            if len(inference_session.hotstart_removed_obj_ids) > 0:
                obj_ids_to_hide.extend(list(inference_session.hotstart_removed_obj_ids))
            if len(obj_ids_to_hide) > 0:
                obj_ids_to_hide_t = torch.tensor(obj_ids_to_hide, dtype=torch.int64)
                keep &= ~torch.isin(out_obj_ids, obj_ids_to_hide_t)

            # slice those valid entries from the original outputs
            keep_idx = torch.nonzero(keep, as_tuple=True)[0]
            keep_idx_gpu = keep_idx.to(device=out_binary_masks.device, non_blocking=True)

            out_obj_ids = torch.index_select(out_obj_ids, 0, keep_idx)
            out_probs = torch.index_select(out_probs, 0, keep_idx)
            out_tracker_probs = torch.index_select(out_tracker_probs, 0, keep_idx)
            out_binary_masks = torch.index_select(out_binary_masks, 0, keep_idx_gpu)

            out_boxes_xyxy = masks_to_boxes(out_binary_masks)

        # Apply non-overlapping constraints on the existing masklets.
        # Constraints are enforced independently per prompt group.
        if out_binary_masks.shape[0] > 1:
            assert len(out_binary_masks) == len(out_tracker_probs)
            prompt_ids_filtered = [
                inference_session.obj_id_to_prompt_id[int(obj_id)] for obj_id in out_obj_ids.tolist()
            ]
            out_binary_masks = (
                self._apply_object_wise_non_overlapping_constraints(
                    out_binary_masks.unsqueeze(1),
                    out_tracker_probs.unsqueeze(1).to(out_binary_masks.device),
                    background_value=0,
                    prompt_ids=prompt_ids_filtered,
                ).squeeze(1)
            ) > 0

        # Build prompt mappings.
        prompt_id_to_obj_ids = {}
        prompt_to_obj_ids = {}
        for obj_id in out_obj_ids.tolist():
            prompt_id = inference_session.obj_id_to_prompt_id[obj_id]
            prompt_id_to_obj_ids.setdefault(prompt_id, []).append(obj_id)
            prompt_text = inference_session.prompts[prompt_id]
            prompt_to_obj_ids.setdefault(prompt_text, []).append(obj_id)

        outputs = {
            "object_ids": out_obj_ids,
            "scores": out_probs,
            "boxes": out_boxes_xyxy,
            "masks": out_binary_masks,
            "prompt_id_to_obj_ids": prompt_id_to_obj_ids,
            "prompt_to_obj_ids": prompt_to_obj_ids,
        }
        return outputs


__all__ = ["Sam3VideoProcessor"]
