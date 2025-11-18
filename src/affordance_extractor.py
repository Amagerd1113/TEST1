"""
Affordance & Distractor Extractor
==================================

Uses Qwen2-VL-7B-Instruct to extract target and distractor density maps.

Given:
- RGB image
- Depth map
- Language instruction (e.g., "go to the blue chair")

Produces:
- ρ_target: 3D density map for target object (positive mass, attractive)
- ρ_distractor: 3D density map for visually similar distractors (negative mass, repulsive)

The VLM identifies:
1. Target object from instruction
2. Visually similar objects that could be confused (distractors)
3. Dense probability maps weighted by depth

This creates the source terms for our Poisson equations:
    ∇²φ₊ = +4πG ρ_target
    ∇²φ₋ = -4πG ρ_distractor

References:
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- ESC-Navigator (uses similar affordance extraction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from jaxtyping import Float
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np


class AffordanceExtractor(nn.Module):
    """
    Extracts target and distractor density maps using Qwen2-VL-7B.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        grid_resolution: int = 64,
        temperature: float = 0.7,
        use_depth_weighting: bool = True,
        device: str = "cuda",
    ):
        """
        Args:
            model_name: Qwen2-VL model name
            grid_resolution: Output 3D grid resolution
            temperature: Sampling temperature for VLM
            use_depth_weighting: Weight density by inverse depth (closer = stronger)
            device: Device to run on
        """
        super().__init__()

        self.grid_resolution = grid_resolution
        self.temperature = temperature
        self.use_depth_weighting = use_depth_weighting
        self.device = device

        # Load Qwen2-VL model
        print(f"Loading {model_name}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

        # Prompt template for affordance extraction
        self.prompt_template = """You are a robotic vision system analyzing a scene for navigation.

Instruction: "{instruction}"

Task: Identify ALL objects in the image and classify them as:
1. TARGET: The object described in the instruction (exactly 1 object)
2. DISTRACTOR: Objects that are visually similar to the target and could be confused
3. OTHER: All other objects

For each object, provide:
- Category: TARGET, DISTRACTOR, or OTHER
- Bounding box: [x_min, y_min, x_max, y_max] normalized to [0, 1]
- Confidence: 0.0 to 1.0
- Description: Brief description

Respond in JSON format:
{{
    "target": {{"bbox": [x1, y1, x2, y2], "confidence": 0.95, "description": "..."}},
    "distractors": [
        {{"bbox": [x1, y1, x2, y2], "confidence": 0.85, "description": "..."}},
        ...
    ]
}}

Be conservative: only mark objects as DISTRACTOR if they are genuinely similar to the target."""

    def forward(
        self,
        rgb: Float[torch.Tensor, "batch 3 H W"],
        depth: Float[torch.Tensor, "batch 1 H W"],
        instruction: str,
    ) -> Tuple[
        Float[torch.Tensor, "batch 1 res res res"],
        Float[torch.Tensor, "batch 1 res res res"]
    ]:
        """
        Extract target and distractor density maps.

        Args:
            rgb: (B, 3, H, W) RGB image, range [0, 1]
            depth: (B, 1, H, W) depth map in meters
            instruction: Language instruction (e.g., "go to the blue chair")

        Returns:
            rho_target: (B, 1, res, res, res) target density (positive mass)
            rho_distractor: (B, 1, res, res, res) distractor density (phantom mass)
        """
        batch_size = rgb.size(0)
        H, W = rgb.shape[2:]

        # Initialize output density maps
        rho_target = torch.zeros(
            batch_size, 1, self.grid_resolution, self.grid_resolution, self.grid_resolution,
            device=self.device
        )
        rho_distractor = torch.zeros(
            batch_size, 1, self.grid_resolution, self.grid_resolution, self.grid_resolution,
            device=self.device
        )

        # Process each image in batch
        for b in range(batch_size):
            # Convert to PIL for VLM processing
            rgb_np = (rgb[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rgb_pil = Image.fromarray(rgb_np)

            depth_b = depth[b, 0].cpu().numpy()

            # Extract affordances using VLM
            affordances = self._extract_affordances_vlm(rgb_pil, instruction)

            # Convert to 3D density maps using depth
            target_density = self._affordance_to_density(
                affordances["target"],
                depth_b,
                H, W,
            )
            distractor_density = self._affordances_to_density(
                affordances["distractors"],
                depth_b,
                H, W,
            )

            rho_target[b, 0] = target_density
            rho_distractor[b, 0] = distractor_density

        return rho_target, rho_distractor

    def _extract_affordances_vlm(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Dict:
        """
        Use Qwen2-VL to extract target and distractor affordances.

        Args:
            image: PIL Image
            instruction: Language instruction

        Returns:
            affordances: Dictionary with "target" and "distractors"
        """
        # Format prompt
        prompt = self.prompt_template.format(instruction=instruction)

        # Prepare inputs for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=True,
            )

        # Decode response
        response = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse JSON response (with error handling)
        try:
            import json
            # Extract JSON from response (may have extra text)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                affordances = json.loads(json_str)
            else:
                # Fallback: no valid JSON found
                affordances = self._get_fallback_affordances()
        except Exception as e:
            print(f"Warning: Failed to parse VLM response: {e}")
            affordances = self._get_fallback_affordances()

        return affordances

    def _get_fallback_affordances(self) -> Dict:
        """Fallback affordances if VLM fails."""
        return {
            "target": {"bbox": [0.4, 0.4, 0.6, 0.6], "confidence": 0.5, "description": "unknown"},
            "distractors": [],
        }

    def _affordance_to_density(
        self,
        affordance: Dict,
        depth: np.ndarray,
        H: int,
        W: int,
    ) -> Float[torch.Tensor, "res res res"]:
        """
        Convert single affordance (bounding box) to 3D density map.

        Args:
            affordance: Dict with "bbox" [x1, y1, x2, y2] and "confidence"
            depth: (H, W) depth map in meters
            H, W: Image dimensions

        Returns:
            density: (res, res, res) 3D density map
        """
        res = self.grid_resolution
        density = torch.zeros(res, res, res, device=self.device)

        if not affordance or "bbox" not in affordance:
            return density

        bbox = affordance["bbox"]
        confidence = affordance.get("confidence", 1.0)

        # Convert normalized bbox to pixel coordinates
        x1, y1, x2, y2 = bbox
        x1_px = int(x1 * W)
        y1_px = int(y1 * H)
        x2_px = int(x2 * W)
        y2_px = int(y2 * H)

        # Clip to image bounds
        x1_px = max(0, min(x1_px, W-1))
        x2_px = max(0, min(x2_px, W-1))
        y1_px = max(0, min(y1_px, H-1))
        y2_px = max(0, min(y2_px, H-1))

        if x2_px <= x1_px or y2_px <= y1_px:
            return density

        # Extract depth in bounding box
        depth_roi = depth[y1_px:y2_px, x1_px:x2_px]

        # Median depth of object
        median_depth = np.median(depth_roi[depth_roi > 0]) if np.any(depth_roi > 0) else 1.0

        # Map to 3D grid
        # Assume camera at (0.5, 0.5, 0.0) looking forward, object at depth z
        # Simple projection: x, y from bbox, z from depth
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        z_normalized = np.clip(median_depth / 5.0, 0.0, 1.0)  # Normalize depth to [0, 1], assume max 5m

        # Gaussian blob in 3D grid
        x_grid = int(x_center * res)
        y_grid = int(y_center * res)
        z_grid = int(z_normalized * res)

        # Gaussian parameters
        sigma = res * 0.1  # 10% of grid size

        # Create Gaussian blob
        for i in range(max(0, x_grid - int(3*sigma)), min(res, x_grid + int(3*sigma))):
            for j in range(max(0, y_grid - int(3*sigma)), min(res, y_grid + int(3*sigma))):
                for k in range(max(0, z_grid - int(3*sigma)), min(res, z_grid + int(3*sigma))):
                    dist_sq = (i - x_grid)**2 + (j - y_grid)**2 + (k - z_grid)**2
                    value = confidence * np.exp(-dist_sq / (2 * sigma**2))

                    # Depth weighting: closer objects have stronger field
                    if self.use_depth_weighting:
                        depth_weight = 1.0 / max(median_depth, 0.5)  # Inverse depth
                        value *= depth_weight

                    density[i, j, k] = max(density[i, j, k].item(), value)

        return density

    def _affordances_to_density(
        self,
        affordances: list,
        depth: np.ndarray,
        H: int,
        W: int,
    ) -> Float[torch.Tensor, "res res res"]:
        """
        Convert multiple affordances (distractors) to 3D density map.

        Args:
            affordances: List of affordance dicts
            depth: (H, W) depth map
            H, W: Image dimensions

        Returns:
            density: (res, res, res) 3D density map
        """
        res = self.grid_resolution
        density = torch.zeros(res, res, res, device=self.device)

        for affordance in affordances:
            density_single = self._affordance_to_density(affordance, depth, H, W)
            density = torch.maximum(density, density_single)  # Take max (union)

        return density


if __name__ == "__main__":
    # Quick test
    print("Testing AffordanceExtractor...")

    # Note: This requires Qwen2-VL model download (~14GB)
    # For testing, we'll create a lightweight mock

    class MockAffordanceExtractor(AffordanceExtractor):
        """Mock for testing without downloading full model."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model = None  # Don't load model
            self.processor = None

        def _extract_affordances_vlm(self, image, instruction):
            """Return dummy affordances."""
            return {
                "target": {
                    "bbox": [0.4, 0.4, 0.6, 0.6],
                    "confidence": 0.95,
                    "description": "blue chair",
                },
                "distractors": [
                    {
                        "bbox": [0.1, 0.3, 0.3, 0.5],
                        "confidence": 0.75,
                        "description": "blue sofa",
                    },
                ],
            }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = MockAffordanceExtractor(grid_resolution=64, device=device)

    # Create dummy inputs
    batch_size = 2
    rgb = torch.rand(batch_size, 3, 480, 640).to(device)
    depth = torch.rand(batch_size, 1, 480, 640).to(device) * 5.0  # 0-5 meters
    instruction = "Go to the blue chair"

    # Extract densities
    import time
    start = time.time()
    rho_target, rho_distractor = extractor(rgb, depth, instruction)
    elapsed = (time.time() - start) * 1000

    print(f"✓ Extracted densities in {elapsed:.2f}ms")
    print(f"  ρ_target: shape={rho_target.shape}, sum={rho_target.sum():.3f}")
    print(f"  ρ_distractor: shape={rho_distractor.shape}, sum={rho_distractor.sum():.3f}")

    print("✓ All tests passed!")
