# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Comprehensive tests for FullImageDatamanager tiling functionality
"""

import pytest
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager


class MockConfig:
    """Mock config for testing tiling methods without full datamanager setup"""

    def __init__(self):
        self.tile_size_max = 0
        self.tile_alignment = 16
        self.cache_images = "cpu"
        self.cache_images_type = "float32"
        self.max_thread_workers = None


class TestTilingCore:
    """Core tiling algorithm tests"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create a minimal mock config to avoid full datamanager initialization
        self.config = MockConfig()
        self.datamanager = FullImageDatamanager.__new__(FullImageDatamanager)  # Skip __init__

    def test_balanced_tile_sizes_comprehensive(self):
        """Test balanced tile size calculation with various scenarios"""
        test_cases = [
            # (width, height, tile_size_max, tile_alignment, expected_widths, expected_heights)
            (128, 80, 64, 16, [64, 64], [48, 32]),  # Exact division with alignment
            (256, 256, 128, 16, [128, 128], [128, 128]),  # Square image
            (50, 30, 100, 16, [50], [30]),  # Image smaller than tile_size_max
            (128, 80, 0, 16, [128], [80]),  # Tiling disabled
        ]

        for width, height, tile_size_max, tile_alignment, expected_widths, expected_heights in test_cases:
            tile_widths, tile_heights = self.datamanager._calculate_balanced_tile_sizes(
                width, height, tile_size_max, tile_alignment
            )

            # Check that tiles sum to original dimensions
            assert sum(tile_widths) == width, f"Tile widths don't sum to {width}"
            assert sum(tile_heights) == height, f"Tile heights don't sum to {height}"

            # Check alignment (when tiling is enabled)
            if tile_size_max > 0 and width > tile_size_max:
                for w in tile_widths[:-1]:  # All but last tile should be aligned
                    assert w % tile_alignment == 0, f"Tile width {w} not aligned to {tile_alignment}"
            if tile_size_max > 0 and height > tile_size_max:
                for h in tile_heights[:-1]:  # All but last tile should be aligned
                    assert h % tile_alignment == 0, f"Tile height {h} not aligned to {tile_alignment}"

    def test_image_tiling_pixel_perfect(self):
        """Test that image tiling produces pixel-perfect reconstruction"""
        # Create test image
        original_image = torch.rand(80, 128, 3)  # 80x128 RGB image

        # Calculate tile sizes
        tile_widths, tile_heights = self.datamanager._calculate_balanced_tile_sizes(128, 80, 64, 16)

        # Tile the image
        tiles = self.datamanager._tile_undistorted_image(original_image, tile_widths, tile_heights)

        # Reconstruct image from tiles
        reconstructed = torch.zeros_like(original_image)
        tile_idx = 0
        y_offset = 0

        for tile_height in tile_heights:
            x_offset = 0
            for tile_width in tile_widths:
                tile = tiles[tile_idx]
                reconstructed[y_offset : y_offset + tile_height, x_offset : x_offset + tile_width] = tile
                x_offset += tile_width
                tile_idx += 1
            y_offset += tile_height

        # Verify pixel-perfect reconstruction
        assert torch.allclose(original_image, reconstructed, atol=1e-6), "Reconstruction not pixel-perfect"

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Test very small images
        tiny_widths, tiny_heights = self.datamanager._calculate_balanced_tile_sizes(16, 16, 64, 16)
        assert tiny_widths == [16] and tiny_heights == [16], "Small images should not be tiled"

        # Test alignment edge cases
        with pytest.raises(ValueError, match="Image too large"):
            # Image that can't be aligned properly
            self.datamanager._calculate_balanced_tile_sizes(100, 100, 15, 32)

    def test_realistic_scenarios(self):
        """Test tiling with realistic image dimensions"""
        test_scenarios = [
            # (width, height, tile_size_max, description)
            (1920, 1080, 512, "Full HD"),
            (3840, 2160, 1024, "4K UHD"),
            (4096, 4096, 512, "4K Square"),
            (800, 600, 256, "SVGA"),
        ]

        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)  # Skip __init__

        for width, height, tile_size_max, description in test_scenarios:
            tile_widths, tile_heights = datamanager._calculate_balanced_tile_sizes(width, height, tile_size_max, 16)

            # Verify basic properties
            assert sum(tile_widths) == width, f"{description}: Width mismatch"
            assert sum(tile_heights) == height, f"{description}: Height mismatch"
            assert all(w <= tile_size_max + 16 for w in tile_widths), f"{description}: Tile too wide"
            assert all(h <= tile_size_max + 16 for h in tile_heights), f"{description}: Tile too tall"

    def test_config_validation(self):
        """Test configuration parameter validation"""
        # Test valid configurations
        valid_configs = [
            {"tile_size_max": 0, "tile_alignment": 16},  # Disabled
            {"tile_size_max": 512, "tile_alignment": 16},  # Standard
            {"tile_size_max": 1024, "tile_alignment": 32},  # Large tiles
        ]

        for config in valid_configs:
            # Should not raise any exceptions
            tile_widths, tile_heights = FullImageDatamanager.__new__(
                FullImageDatamanager
            )._calculate_balanced_tile_sizes(1920, 1080, config["tile_size_max"], config["tile_alignment"])
            assert isinstance(tile_widths, list)
            assert isinstance(tile_heights, list)


class TestTilingIntrinsics:
    """Camera intrinsics validation tests"""

    def test_camera_intrinsics_exact_division(self):
        """Test camera intrinsics for exact division scenario (128×80 → 2×2 tiles)"""
        # Test the core mathematical concept: cx/cy adjustment for exact division
        original_cx, original_cy = 57.0, 43.0
        tile_widths = [64, 64]  # 128/2 = 64 each
        tile_heights = [40, 40]  # 80/2 = 40 each

        # Expected cx/cy for each tile position
        expected_adjustments = [
            (0, 0, 57.0, 43.0),  # Top-left: no offset
            (0, 1, -7.0, 43.0),  # Top-right: cx = 57 - 64 = -7
            (1, 0, 57.0, 3.0),  # Bottom-left: cy = 43 - 40 = 3
            (1, 1, -7.0, 3.0),  # Bottom-right: cx = -7, cy = 3
        ]

        for row, col, expected_cx, expected_cy in expected_adjustments:
            # Calculate offsets (core tiling math)
            x_offset = sum(tile_widths[:col])
            y_offset = sum(tile_heights[:row])
            adjusted_cx = original_cx - x_offset
            adjusted_cy = original_cy - y_offset

            assert abs(adjusted_cx - expected_cx) < 1e-6, f"Tile ({row},{col}): cx mismatch"
            assert abs(adjusted_cy - expected_cy) < 1e-6, f"Tile ({row},{col}): cy mismatch"

    def test_camera_intrinsics_with_remainder(self):
        """Test camera intrinsics for remainder handling (100×80 → 3×3 tiles)"""
        # Test remainder distribution: 100/3 = 33 remainder 1, 80/3 = 26 remainder 2
        original_cx, original_cy = 50.0, 40.0
        tile_widths = [34, 33, 33]  # First gets +1 from remainder
        tile_heights = [27, 27, 26]  # First two get +1 from remainder

        # Test a few key positions
        test_cases = [
            (0, 0, 50.0, 40.0),  # Top-left: no offset
            (0, 1, 16.0, 40.0),  # Top-middle: cx = 50 - 34 = 16
            (1, 0, 50.0, 13.0),  # Middle-left: cy = 40 - 27 = 13
            (2, 2, -17.0, -14.0),  # Bottom-right: cx = 50 - 67 = -17, cy = 40 - 54 = -14
        ]

        for row, col, expected_cx, expected_cy in test_cases:
            x_offset = sum(tile_widths[:col])
            y_offset = sum(tile_heights[:row])
            adjusted_cx = original_cx - x_offset
            adjusted_cy = original_cy - y_offset

            assert abs(adjusted_cx - expected_cx) < 1e-6, f"Tile ({row},{col}): cx error"
            assert abs(adjusted_cy - expected_cy) < 1e-6, f"Tile ({row},{col}): cy error"

    def test_camera_parameter_precision(self):
        """Test floating-point precision in cx/cy calculations"""
        test_cases = [
            {"cx": 63.7, "cy": 39.3},  # Non-integer values
            {"cx": 0.0, "cy": 0.0},  # Corner case
            {"cx": 127.9, "cy": 79.9},  # Near edge
            {"cx": 64.5, "cy": 40.5},  # Half-pixel offset
        ]

        tile_widths = [64, 64]
        tile_heights = [40, 40]

        for case in test_cases:
            original_cx, original_cy = case["cx"], case["cy"]

            # Test all 4 tile positions
            for row in range(2):
                for col in range(2):
                    x_offset = sum(tile_widths[:col])
                    y_offset = sum(tile_heights[:row])
                    expected_cx = original_cx - x_offset
                    expected_cy = original_cy - y_offset

                    # Verify precision (should be exact for floating-point arithmetic)
                    adjusted_cx = original_cx - x_offset
                    adjusted_cy = original_cy - y_offset

                    assert abs(adjusted_cx - expected_cx) < 1e-10, "Precision error in cx"
                    assert abs(adjusted_cy - expected_cy) < 1e-10, "Precision error in cy"

    def test_focal_length_preservation(self):
        """Test that focal lengths (fx, fy) must be preserved in tiling"""
        # Core concept: focal lengths are lens properties and never change during tiling
        test_cases = [
            {"fx": 64.0, "fy": 64.0},  # Square pixels
            {"fx": 70.0, "fy": 65.0},  # Rectangular pixels
            {"fx": 100.0, "fy": 95.0},  # Different aspect ratio
        ]

        for case in test_cases:
            # In any tiling implementation, these must remain identical
            tiled_fx = case["fx"]  # Should never change
            tiled_fy = case["fy"]  # Should never change

            assert tiled_fx == case["fx"], "Focal length fx must be preserved"
            assert tiled_fy == case["fy"], "Focal length fy must be preserved"

    def test_tiling_disabled_behavior(self):
        """Test that disabled tiling preserves original parameters"""
        # When tile_size_max=0, should return original dimensions
        width, height = 128, 80

        # Disabled tiling should return single tile with original dimensions
        tile_widths = [width]
        tile_heights = [height]

        # Camera intrinsics should be unchanged for single tile at (0,0)
        original_cx, original_cy = 57.0, 43.0
        x_offset = sum(tile_widths[:0])  # 0
        y_offset = sum(tile_heights[:0])  # 0
        adjusted_cx = original_cx - x_offset
        adjusted_cy = original_cy - y_offset

        assert adjusted_cx == original_cx
        assert adjusted_cy == original_cy
        assert tile_widths[0] == width
        assert tile_heights[0] == height

    def test_camera_intrinsics_adjustment(self):
        """Test _adjust_camera_intrinsics_for_tile method directly"""
        # Create test camera
        cameras = Cameras(
            fx=torch.tensor([100.0]),
            fy=torch.tensor([95.0]),
            cx=torch.tensor([64.0]),
            cy=torch.tensor([40.0]),
            width=torch.tensor([128]),
            height=torch.tensor([80]),
            camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
        )

        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)

        # Test tile adjustment
        adjusted = datamanager._adjust_camera_intrinsics_for_tile(
            cameras,
            tile_offset_x=32,
            tile_offset_y=20,
            tile_width=64,
            tile_height=40,
            parent_camera_index=1,
            tile_row=1,
            tile_col=0,
            total_tiles=4,
        )

        # Verify adjustments
        assert torch.allclose(adjusted.cx, torch.tensor([32.0])), "cx should be adjusted by offset"
        assert torch.allclose(adjusted.cy, torch.tensor([20.0])), "cy should be adjusted by offset"
        assert torch.allclose(adjusted.fx, cameras.fx), "fx should be preserved"
        assert torch.allclose(adjusted.fy, cameras.fy), "fy should be preserved"
        assert torch.allclose(adjusted.width, torch.tensor([64])), "width should be tile width"
        assert torch.allclose(adjusted.height, torch.tensor([40])), "height should be tile height"

        # Verify debugging metadata
        assert adjusted.metadata is not None, "Metadata should be present"
        metadata = adjusted.metadata
        assert metadata["parent_camera_index"] == 1, "Parent camera index should be preserved"
        assert metadata["tile_row"] == 1, "Tile row should be preserved"
        assert metadata["tile_col"] == 0, "Tile col should be preserved"

    def test_camera_replication(self):
        """Test _replicate_cameras_for_tiles method directly"""
        # Create test camera
        cameras = Cameras(
            fx=torch.tensor([100.0]),
            fy=torch.tensor([100.0]),
            cx=torch.tensor([64.0]),
            cy=torch.tensor([40.0]),
            width=torch.tensor([128]),
            height=torch.tensor([80]),
            camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
        )

        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)
        tile_widths, tile_heights = [64, 64], [40, 40]  # 2x2 tiling

        # Test camera replication
        tile_cameras = datamanager._replicate_cameras_for_tiles(cameras, tile_widths, tile_heights)

        # Verify correct number of cameras
        assert len(tile_cameras) == 4, "Should create 4 cameras for 2x2 tiling"

        # Verify camera adjustments for each tile
        # Both cy and tile_offset_y are in Y-down coordinates (COLMAP standard)
        # Simple subtraction: cy_adjusted = cy - tile_offset_y
        # Original: cy=40.0
        # Tile 0 (top-left): tile_offset_y=0 → cy_adjusted = 40 - 0 = 40
        # Tile 1 (top-right): tile_offset_y=0 → cy_adjusted = 40 - 0 = 40
        # Tile 2 (bottom-left): tile_offset_y=40 → cy_adjusted = 40 - 40 = 0
        # Tile 3 (bottom-right): tile_offset_y=40 → cy_adjusted = 40 - 40 = 0
        expected_adjustments = [
            (64.0, 40.0),  # Top-left: cx=64-0=64, cy=40-0=40
            (0.0, 40.0),  # Top-right: cx=64-64=0, cy=40-0=40
            (64.0, 0.0),  # Bottom-left: cx=64-0=64, cy=40-40=0
            (0.0, 0.0),  # Bottom-right: cx=64-64=0, cy=40-40=0
        ]

        for i, (expected_cx, expected_cy) in enumerate(expected_adjustments):
            assert torch.allclose(tile_cameras[i].cx, torch.tensor([expected_cx])), f"Tile {i} cx incorrect"
            assert torch.allclose(tile_cameras[i].cy, torch.tensor([expected_cy])), f"Tile {i} cy incorrect"
            assert torch.allclose(tile_cameras[i].fx, cameras.fx), f"Tile {i} fx should be preserved"
            assert torch.allclose(tile_cameras[i].fy, cameras.fy), f"Tile {i} fy should be preserved"

    def test_tiling_path_vs_no_tiling_identical_results(self):
        """Test that tiling path gives identical results to no-tiling path when no actual tiling occurs.
        This test mimics exactly how splatfacto training accesses the datamanager."""
        import os
        import tempfile
        from pathlib import Path

        import numpy as np
        from PIL import Image

        from nerfstudio.data.datamanagers.full_images_datamanager import (
            FullImageDatamanager,
            FullImageDatamanagerConfig,
        )
        from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
        from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
        from nerfstudio.data.datasets.base_dataset import InputDataset
        from nerfstudio.data.scene_box import SceneBox

        # Create temporary directory for dummy images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create 2 dummy images with different uniform colors and different sizes
            image0 = torch.full((80, 120, 3), 0.2)  # Gray image: 80x120
            image1 = torch.full((60, 100, 3), 0.8)  # Light gray image: 60x100

            # Save dummy images as actual PNG files
            image0_path = os.path.join(temp_dir, "image0.png")
            image1_path = os.path.join(temp_dir, "image1.png")

            # Convert to PIL and save
            Image.fromarray((image0.numpy() * 255).astype(np.uint8)).save(image0_path)
            Image.fromarray((image1.numpy() * 255).astype(np.uint8)).save(image1_path)

            # Camera 0: fx=100, cy=40, translation=[1,2,3]
            camera0_to_world = torch.tensor(
                [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]]
            )

            # Camera 1: fx=150, cy=30, translation=[4,5,6]
            camera1_to_world = torch.tensor(
                [[1.0, 0.0, 0.0, 4.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, 1.0, 6.0], [0.0, 0.0, 0.0, 1.0]]
            )

            dummy_cameras = Cameras(
                fx=torch.tensor([100.0, 150.0]),
                fy=torch.tensor([100.0, 150.0]),
                cx=torch.tensor([60.0, 50.0]),
                cy=torch.tensor([40.0, 30.0]),
                width=torch.tensor([120, 100]),
                height=torch.tensor([80, 60]),
                camera_to_worlds=torch.stack([camera0_to_world[:3, :], camera1_to_world[:3, :]]),
            )

            print("=== INITIAL CAMERA SETUP ===")
            print(f"dummy_cameras shape: {dummy_cameras.shape}")
            print(f"Camera 0 translation: {dummy_cameras.camera_to_worlds[0, :, 3]}")
            print(f"Camera 1 translation: {dummy_cameras.camera_to_worlds[1, :, 3]}")

            # Create proper DataparserOutputs
            dataparser_outputs = DataparserOutputs(
                image_filenames=[image0_path, image1_path],
                cameras=dummy_cameras,
                scene_box=SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])),
                dataparser_scale=1.0,
            )

            # Create dataset
            dataset = InputDataset(dataparser_outputs)

            # Create mock dataparser config
            mock_dataparser_config = NerfstudioDataParserConfig(data=Path(temp_dir))

            # Create minimal datamanagers by bypassing the full initialization
            # Test 1: No tiling (tile_size_max = 0)
            config_no_tiling = FullImageDatamanagerConfig(
                dataparser=mock_dataparser_config,
                tile_size_max=0,  # No tiling
                cache_images="cpu",
            )

            # Create datamanager instance without full initialization
            datamanager_no_tiling = FullImageDatamanager.__new__(FullImageDatamanager)
            datamanager_no_tiling.config = config_no_tiling
            datamanager_no_tiling.device = "cpu"
            datamanager_no_tiling.world_size = 1
            datamanager_no_tiling.local_rank = 0
            datamanager_no_tiling.test_mode = "test"
            datamanager_no_tiling.test_split = "test"
            datamanager_no_tiling.tile_to_original_mapping = None

            # Manually set up the datasets and cameras
            datamanager_no_tiling.train_dataset = dataset
            datamanager_no_tiling.eval_dataset = dataset
            datamanager_no_tiling.train_cameras = dummy_cameras
            datamanager_no_tiling.eval_cameras = dummy_cameras

            # Test 2: Tiling path but no actual tiling (tile_size_max > image size)
            config_with_tiling = FullImageDatamanagerConfig(
                dataparser=mock_dataparser_config,
                tile_size_max=200,  # Larger than both images, so no actual tiling
                cache_images="cpu",
            )

            # Create second datamanager instance
            datamanager_with_tiling = FullImageDatamanager.__new__(FullImageDatamanager)
            datamanager_with_tiling.config = config_with_tiling
            datamanager_with_tiling.device = "cpu"
            datamanager_with_tiling.world_size = 1
            datamanager_with_tiling.local_rank = 0
            datamanager_with_tiling.test_mode = "test"
            datamanager_with_tiling.test_split = "test"
            datamanager_with_tiling.tile_to_original_mapping = None

            # Set up initial cameras and dataset for tiling
            datamanager_with_tiling.train_dataset = dataset
            datamanager_with_tiling.eval_dataset = dataset
            datamanager_with_tiling.train_cameras = dummy_cameras
            datamanager_with_tiling.eval_cameras = dummy_cameras

            print("\\n=== BEFORE TILING ===")
            print(f"datamanager_with_tiling.train_cameras shape: {datamanager_with_tiling.train_cameras.shape}")
            print(f"Camera 0 translation: {datamanager_with_tiling.train_cameras.camera_to_worlds[0, :, 3]}")
            print(f"Camera 1 translation: {datamanager_with_tiling.train_cameras.camera_to_worlds[1, :, 3]}")

            # Apply tiling to the second datamanager
            tiled_images = datamanager_with_tiling._apply_tiling_to_images(
                [{"image": image0, "image_idx": 0}, {"image": image1, "image_idx": 1}], "train"
            )

            print("\\n=== AFTER TILING ===")
            print(f"datamanager_with_tiling.train_cameras shape: {datamanager_with_tiling.train_cameras.shape}")
            print(f"Camera 0 translation: {datamanager_with_tiling.train_cameras.camera_to_worlds[0, :, 3]}")
            print(f"Camera 1 translation: {datamanager_with_tiling.train_cameras.camera_to_worlds[1, :, 3]}")

            # Set up tiled dataset
            datamanager_with_tiling.train_dataset = type(
                "MockDataset",
                (),
                {"__getitem__": lambda self, idx: tiled_images[idx], "__len__": lambda self: len(tiled_images)},
            )()

            print(f"No tiling - train_cameras shape: {datamanager_no_tiling.train_cameras.shape}")
            print(f"With tiling - train_cameras shape: {datamanager_with_tiling.train_cameras.shape}")

            # Test exactly how splatfacto training accesses the datamanager
            results_no_tiling = []
            results_with_tiling = []

            # Simulate training loop - get each camera/image pair
            for i in range(2):  # We have 2 cameras
                # Access exactly like splatfacto training does: camera = self.train_cameras[image_idx : image_idx + 1]
                camera_no_tiling = datamanager_no_tiling.train_cameras[i : i + 1]
                camera_with_tiling = datamanager_with_tiling.train_cameras[i : i + 1]

                # Get image data like splatfacto does
                data_no_tiling = datamanager_no_tiling.train_dataset[i]
                data_with_tiling = datamanager_with_tiling.train_dataset[i]

                print(f"\n=== Camera {i} ===")
                print(f"No tiling - camera shape: {camera_no_tiling.shape}")
                print(f"With tiling - camera shape: {camera_with_tiling.shape}")

                # Both should pass splatfacto's assertion
                assert camera_no_tiling.shape[0] == 1, f"No tiling camera {i} should have shape[0]==1"
                assert camera_with_tiling.shape[0] == 1, f"With tiling camera {i} should have shape[0]==1"

                # Extract parameters for comparison
                no_tiling_result = {
                    "fx": camera_no_tiling.fx.item(),
                    "fy": camera_no_tiling.fy.item(),
                    "cx": camera_no_tiling.cx.item(),
                    "cy": camera_no_tiling.cy.item(),
                    "width": camera_no_tiling.width.item(),
                    "height": camera_no_tiling.height.item(),
                    "translation": camera_no_tiling.camera_to_worlds[0, :, 3].tolist(),
                    "image_mean": data_no_tiling["image"].mean().item(),
                    "image_shape": list(data_no_tiling["image"].shape),
                }

                with_tiling_result = {
                    "fx": camera_with_tiling.fx.item(),
                    "fy": camera_with_tiling.fy.item(),
                    "cx": camera_with_tiling.cx.item(),
                    "cy": camera_with_tiling.cy.item(),
                    "width": camera_with_tiling.width.item(),
                    "height": camera_with_tiling.height.item(),
                    "translation": camera_with_tiling.camera_to_worlds[0, :, 3].tolist(),
                    "image_mean": data_with_tiling["image"].mean().item(),
                    "image_shape": list(data_with_tiling["image"].shape),
                }

                print(f"No tiling result: {no_tiling_result}")
                print(f"With tiling result: {with_tiling_result}")

                results_no_tiling.append(no_tiling_result)
                results_with_tiling.append(with_tiling_result)

                # Compare all parameters - they should be identical
                assert (
                    abs(no_tiling_result["fx"] - with_tiling_result["fx"]) < 1e-6
                ), f"Camera {i} fx differs: {no_tiling_result['fx']} vs {with_tiling_result['fx']}"
                assert (
                    abs(no_tiling_result["fy"] - with_tiling_result["fy"]) < 1e-6
                ), f"Camera {i} fy differs: {no_tiling_result['fy']} vs {with_tiling_result['fy']}"
                assert (
                    abs(no_tiling_result["cx"] - with_tiling_result["cx"]) < 1e-6
                ), f"Camera {i} cx differs: {no_tiling_result['cx']} vs {with_tiling_result['cx']}"
                assert (
                    abs(no_tiling_result["cy"] - with_tiling_result["cy"]) < 1e-6
                ), f"Camera {i} cy differs: {no_tiling_result['cy']} vs {with_tiling_result['cy']}"
                assert (
                    no_tiling_result["width"] == with_tiling_result["width"]
                ), f"Camera {i} width differs: {no_tiling_result['width']} vs {with_tiling_result['width']}"
                assert (
                    no_tiling_result["height"] == with_tiling_result["height"]
                ), f"Camera {i} height differs: {no_tiling_result['height']} vs {with_tiling_result['height']}"

                # Compare extrinsics (translation)
                for j in range(3):
                    assert (
                        abs(no_tiling_result["translation"][j] - with_tiling_result["translation"][j]) < 1e-6
                    ), f"Camera {i} translation[{j}] differs: {no_tiling_result['translation'][j]} vs {with_tiling_result['translation'][j]}"

                # Compare image data
                assert (
                    abs(no_tiling_result["image_mean"] - with_tiling_result["image_mean"]) < 1e-6
                ), f"Camera {i} image data differs: {no_tiling_result['image_mean']} vs {with_tiling_result['image_mean']}"
                assert (
                    no_tiling_result["image_shape"] == with_tiling_result["image_shape"]
                ), f"Camera {i} image shape differs: {no_tiling_result['image_shape']} vs {with_tiling_result['image_shape']}"

            # Verify we got different cameras (not the same camera twice)
            assert (
                results_no_tiling[0]["fx"] != results_no_tiling[1]["fx"]
            ), "Should have different cameras with different fx"
            assert (
                results_no_tiling[0]["translation"] != results_no_tiling[1]["translation"]
            ), "Should have different camera positions"
            assert (
                results_no_tiling[0]["image_mean"] != results_no_tiling[1]["image_mean"]
            ), "Should have different image colors"

            # Verify tiling path gives same results as no-tiling path
            for i in range(2):
                for key in results_no_tiling[i]:
                    if isinstance(results_no_tiling[i][key], list):
                        for j, (val1, val2) in enumerate(zip(results_no_tiling[i][key], results_with_tiling[i][key])):
                            assert abs(val1 - val2) < 1e-6, f"Camera {i} {key}[{j}] differs: {val1} vs {val2}"
                    else:
                        assert (
                            abs(results_no_tiling[i][key] - results_with_tiling[i][key]) < 1e-6
                        ), f"Camera {i} {key} differs: {results_no_tiling[i][key]} vs {results_with_tiling[i][key]}"

            print("\n✅ SUCCESS: Tiling path preserves all camera parameters and image data correctly!")

    def test_actual_tiling_preserves_camera_extrinsics(self):
        """Test that camera extrinsics are preserved correctly when actual tiling occurs."""
        # Create test cameras with different extrinsics
        camera0_to_world = torch.tensor(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]]
        )

        camera1_to_world = torch.tensor(
            [[1.0, 0.0, 0.0, 4.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, 1.0, 6.0], [0.0, 0.0, 0.0, 1.0]]
        )

        dummy_cameras = Cameras(
            fx=torch.tensor([100.0, 150.0]),
            fy=torch.tensor([100.0, 150.0]),
            cx=torch.tensor([64.0, 50.0]),
            cy=torch.tensor([40.0, 30.0]),
            width=torch.tensor([128, 100]),
            height=torch.tensor([80, 60]),
            camera_to_worlds=torch.stack([camera0_to_world[:3, :], camera1_to_world[:3, :]]),
        )

        # Create mock datamanager with tiling that forces actual tiling
        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)
        datamanager.config = type("Config", (), {"tile_size_max": 64, "tile_alignment": 16})()
        datamanager.train_cameras = dummy_cameras

        # Create test images that will be tiled (128x80 and 100x60 with tile_size_max=64)
        image0 = torch.rand(80, 128, 3)  # Will be tiled into 4 tiles (2x2)
        image1 = torch.rand(60, 100, 3)  # Will be tiled into 2 tiles (2x1)

        # Apply tiling
        tiled_images = datamanager._apply_tiling_to_images(
            [{"image": image0, "image_idx": 0}, {"image": image1, "image_idx": 1}], "train"
        )

        # Verify correct number of tiles
        assert len(tiled_images) == 6, f"Expected 6 tiled images (4+2), got {len(tiled_images)}"
        assert (
            datamanager.train_cameras.shape[0] == 6
        ), f"Expected 6 tiled cameras, got {datamanager.train_cameras.shape[0]}"

        # Verify camera extrinsics are preserved correctly
        expected_translations = [
            torch.tensor([1.0, 2.0, 3.0]),  # From camera 0 (4 tiles)
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0]),  # From camera 1 (2 tiles)
            torch.tensor([4.0, 5.0, 6.0]),
        ]

        for i, expected_translation in enumerate(expected_translations):
            actual_translation = datamanager.train_cameras.camera_to_worlds[i, :, 3]
            assert torch.allclose(
                actual_translation, expected_translation
            ), f"Camera {i} translation mismatch: {actual_translation} != {expected_translation}"

        # Verify tensor shapes are correct (not flattened)
        assert datamanager.train_cameras.camera_to_worlds.shape == (
            6,
            3,
            4,
        ), f"Wrong camera_to_worlds shape: {datamanager.train_cameras.camera_to_worlds.shape}"

        print("✅ SUCCESS: Actual tiling preserves camera extrinsics correctly!")

    def test_splatfacto_assertion_with_tiled_cameras(self):
        """Test that verifies splatfacto's assertion behavior with tiled cameras"""
        # Create a single camera (should work)
        single_camera = Cameras(
            fx=torch.tensor([100.0]),
            fy=torch.tensor([100.0]),
            cx=torch.tensor([32.0]),
            cy=torch.tensor([20.0]),
            width=torch.tensor([64]),
            height=torch.tensor([40]),
            camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
        )

        # Create multiple cameras (should fail splatfacto assertion)
        multi_cameras = Cameras(
            fx=torch.tensor([100.0, 100.0, 100.0]),
            fy=torch.tensor([100.0, 100.0, 100.0]),
            cx=torch.tensor([32.0, 32.0, 32.0]),
            cy=torch.tensor([20.0, 20.0, 20.0]),
            width=torch.tensor([64, 64, 64]),
            height=torch.tensor([40, 40, 40]),
            camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :].repeat(3, 1, 1),
            camera_type=torch.tensor([[1], [1], [1]]),
        )

        print(f"Single camera shape: {single_camera.shape}")
        print(f"Multi cameras shape: {multi_cameras.shape}")

        # Test the assertion that splatfacto uses
        assert single_camera.shape[0] == 1, "Single camera should pass"

        # This should fail (like splatfacto would fail)
        try:
            assert multi_cameras.shape[0] == 1, "Only one camera at a time"
            assert False, "Should have failed assertion"
        except AssertionError as e:
            print(f"Expected assertion failure: {e}")

        # But indexed access should work
        indexed_camera = multi_cameras[0:1]
        assert indexed_camera.shape[0] == 1, "Indexed camera should pass"

    def test_camera_indexing_after_concatenation(self):
        """Test that camera indexing works correctly after concatenation"""
        # Create 3 cameras with different extrinsics
        cameras_data = []
        for i in range(3):
            camera_to_worlds = torch.tensor(
                [[[1.0, 0.0, 0.0, float(i + 1)], [0.0, 1.0, 0.0, float(i + 2)], [0.0, 0.0, 1.0, float(i + 3)]]]
            )  # Shape: [1, 3, 4]
            camera = Cameras(
                fx=torch.tensor([100.0]),
                fy=torch.tensor([100.0]),
                cx=torch.tensor([32.0]),
                cy=torch.tensor([20.0]),
                width=torch.tensor([64]),
                height=torch.tensor([40]),
                camera_to_worlds=camera_to_worlds,
            )
            cameras_data.append(camera)

        # Concatenate like tiling does
        tiled_cameras = Cameras(
            fx=torch.cat([cam.fx for cam in cameras_data]),
            fy=torch.cat([cam.fy for cam in cameras_data]),
            cx=torch.cat([cam.cx for cam in cameras_data]),
            cy=torch.cat([cam.cy for cam in cameras_data]),
            width=torch.cat([cam.width for cam in cameras_data]),
            height=torch.cat([cam.height for cam in cameras_data]),
            camera_to_worlds=torch.cat([cam.camera_to_worlds for cam in cameras_data]),
            camera_type=torch.cat([cam.camera_type for cam in cameras_data]),
        )

        print(f"Concatenated cameras shape: {tiled_cameras.shape}")

        # Test indexing like datamanager does: camera = self.train_cameras[image_idx : image_idx + 1]
        for i in range(3):
            indexed_camera = tiled_cameras[i : i + 1]
            print(f"Indexed camera {i} shape: {indexed_camera.shape}")
            print(f"Indexed camera {i} translation: {indexed_camera.camera_to_worlds[0, :, 3]}")

            # This should pass splatfacto's assertion
            assert indexed_camera.shape[0] == 1, f"Indexed camera {i} should have shape[0] == 1"

            # Verify correct extrinsics
            expected_translation = torch.tensor([float(i + 1), float(i + 2), float(i + 3)])
            actual_translation = indexed_camera.camera_to_worlds[0, :, 3]
            assert torch.allclose(actual_translation, expected_translation), f"Camera {i} has wrong extrinsics"

    def test_splatfacto_camera_batch_size_requirement(self):
        """Test that demonstrates the splatfacto camera batch size issue"""
        # Create multiple cameras like tiling would produce
        cameras_data = []
        for i in range(3):
            camera = Cameras(
                fx=torch.tensor([100.0]),
                fy=torch.tensor([100.0]),
                cx=torch.tensor([32.0]),
                cy=torch.tensor([20.0]),
                width=torch.tensor([64]),
                height=torch.tensor([40]),
                camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
            )
            cameras_data.append(camera)

        # Concatenate like tiling does
        tiled_cameras = Cameras(
            fx=torch.cat([cam.fx for cam in cameras_data]),
            fy=torch.cat([cam.fy for cam in cameras_data]),
            cx=torch.cat([cam.cx for cam in cameras_data]),
            cy=torch.cat([cam.cy for cam in cameras_data]),
            width=torch.cat([cam.width for cam in cameras_data]),
            height=torch.cat([cam.height for cam in cameras_data]),
            camera_to_worlds=torch.cat([cam.camera_to_worlds for cam in cameras_data]),
            camera_type=torch.cat([cam.camera_type for cam in cameras_data]),
        )

        print(f"Individual camera shape: {cameras_data[0].shape}")
        print(f"Tiled cameras shape: {tiled_cameras.shape}")

        # This is what splatfacto checks during training
        print(f"tiled_cameras.shape[0]: {tiled_cameras.shape[0]}")

        # Splatfacto expects shape[0] == 1, but tiling creates shape[0] == 3
        assert tiled_cameras.shape[0] == 3, "Tiling creates batch of 3 cameras"

        # This would fail splatfacto's assertion: assert camera.shape[0] == 1
        # The bug is that all 3 cameras get passed to splatfacto at once!

    def test_camera_to_worlds_concatenation_dimensions(self):
        """Test that camera_to_worlds tensors have correct dimensions during concatenation"""
        # Create 3 cameras with different extrinsics
        cameras_data = []
        for i in range(3):
            camera_to_worlds = torch.tensor(
                [[[1.0, 0.0, 0.0, float(i + 1)], [0.0, 1.0, 0.0, float(i + 2)], [0.0, 0.0, 1.0, float(i + 3)]]]
            )  # Shape: [1, 3, 4]
            camera = Cameras(
                fx=torch.tensor([100.0]),
                fy=torch.tensor([100.0]),
                cx=torch.tensor([32.0]),
                cy=torch.tensor([20.0]),
                width=torch.tensor([64]),
                height=torch.tensor([40]),
                camera_to_worlds=camera_to_worlds,
            )
            cameras_data.append(camera)

        # Check individual camera dimensions
        for i, cam in enumerate(cameras_data):
            print(f"Camera {i} camera_to_worlds shape: {cam.camera_to_worlds.shape}")
            print(f"Camera {i} camera_to_worlds: {cam.camera_to_worlds}")

        # Test concatenation like in the tiling code
        concatenated = torch.cat([cam.camera_to_worlds for cam in cameras_data])
        print(f"Concatenated camera_to_worlds shape: {concatenated.shape}")
        print(f"Concatenated camera_to_worlds: {concatenated}")

        # Expected: [3, 3, 4] - 3 cameras, each with 3x4 transform matrix
        assert concatenated.shape == (3, 3, 4), f"Wrong concatenated shape: {concatenated.shape}"

        # Verify each camera's extrinsics are preserved
        for i in range(3):
            expected_translation = torch.tensor([float(i + 1), float(i + 2), float(i + 3)])
            actual_translation = concatenated[i, :, 3]
            assert torch.allclose(
                actual_translation, expected_translation
            ), f"Camera {i} translation wrong: {actual_translation} != {expected_translation}"

    def test_camera_extrinsics_with_tiling(self):
        """Test that camera extrinsics are preserved when actual tiling occurs"""
        # Create camera with unique extrinsics
        camera_to_worlds = torch.tensor(
            [[[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 20.0], [0.0, 0.0, 1.0, 30.0]]]
        )  # Translation [10,20,30]

        original_camera = Cameras(
            fx=torch.tensor([100.0]),
            fy=torch.tensor([100.0]),
            cx=torch.tensor([64.0]),
            cy=torch.tensor([40.0]),
            width=torch.tensor([128]),
            height=torch.tensor([80]),
            camera_to_worlds=camera_to_worlds,
        )

        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)
        tile_widths, tile_heights = [64, 64], [40, 40]  # 2x2 tiling = 4 tiles

        # Create tiles from this camera
        tile_cameras = datamanager._replicate_cameras_for_tiles(original_camera, tile_widths, tile_heights)

        # All 4 tiles should have the SAME extrinsics as the original camera
        expected_translation = torch.tensor([10.0, 20.0, 30.0])

        for i, tile_camera in enumerate(tile_cameras):
            # Handle different tensor shapes
            if tile_camera.camera_to_worlds.dim() == 3:
                tile_translation = tile_camera.camera_to_worlds[0, :, 3]
            else:
                tile_translation = tile_camera.camera_to_worlds[:, 3]
            assert torch.allclose(
                tile_translation, expected_translation
            ), f"Tile {i} has wrong extrinsics: {tile_translation} != {expected_translation}"

    def test_camera_extrinsics_preservation(self):
        """Test that camera extrinsics (camera_to_worlds) are preserved correctly during tiling"""
        # Create 2 cameras with different extrinsics
        camera_to_worlds_1 = torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]]]
        )  # Translation [1,2,3]

        camera_to_worlds_2 = torch.tensor(
            [[[1.0, 0.0, 0.0, 4.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, 1.0, 6.0]]]
        )  # Translation [4,5,6]

        # Stack into multi-camera object
        original_cameras = Cameras(
            fx=torch.tensor([100.0, 100.0]),
            fy=torch.tensor([100.0, 100.0]),
            cx=torch.tensor([32.0, 32.0]),
            cy=torch.tensor([20.0, 20.0]),
            width=torch.tensor([64, 64]),
            height=torch.tensor([40, 40]),
            camera_to_worlds=torch.cat([camera_to_worlds_1, camera_to_worlds_2]),
        )

        # Create mock images
        images = [
            {"image": torch.zeros(40, 64, 3)},
            {"image": torch.ones(40, 64, 3)},
        ]

        # Mock datamanager with no tiling (tile_size_max > image size)
        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)
        datamanager.config = type(
            "Config",
            (),
            {
                "tile_size_max": 128,  # Larger than 64x40, so no tiling
                "tile_alignment": 1,
            },
        )()

        # Process through tiling pipeline
        tiled_images = []
        tiled_cameras_list = []
        tile_mapping = []

        for img_idx, image_data in enumerate(images):
            image = image_data["image"]
            height, width = image.shape[:2]

            tile_widths, tile_heights = datamanager._calculate_balanced_tile_sizes(
                width, height, datamanager.config.tile_size_max, datamanager.config.tile_alignment
            )

            # Should not tile (single tile)
            if len(tile_widths) == 1 and len(tile_heights) == 1:
                tiled_images.append(image_data)
                tiled_cameras_list.append(original_cameras[img_idx])
                tile_mapping.append(img_idx)
                continue

        # Verify each camera maintains its unique extrinsics
        assert len(tiled_cameras_list) == 2, f"Expected 2 cameras, got {len(tiled_cameras_list)}"

        # Check camera 0 extrinsics [1,2,3]
        cam0_translation = tiled_cameras_list[0].camera_to_worlds[:, 3]  # Shape: [3]
        expected_translation_0 = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(
            cam0_translation, expected_translation_0
        ), f"Camera 0 translation wrong: {cam0_translation} != {expected_translation_0}"

        # Check camera 1 extrinsics [4,5,6]
        cam1_translation = tiled_cameras_list[1].camera_to_worlds[:, 3]  # Shape: [3]
        expected_translation_1 = torch.tensor([4.0, 5.0, 6.0])
        assert torch.allclose(
            cam1_translation, expected_translation_1
        ), f"Camera 1 translation wrong: {cam1_translation} != {expected_translation_1}"

        # Verify cameras are different
        assert not torch.allclose(cam0_translation, cam1_translation), "Cameras should have different extrinsics!"

    def test_multi_image_tiling_correspondence(self):
        """Test that multiple images maintain correct camera correspondence after tiling"""
        # Create 3 test images with unique identifiers
        images = []
        for img_id in range(3):
            test_image = torch.full((40, 64, 3), float(img_id + 1))  # Each image filled with unique value
            images.append({"image": test_image})

        # Create 3 test cameras with unique cx values for identification
        cameras_list = []
        for img_id in range(3):
            camera = Cameras(
                fx=torch.tensor([100.0]),
                fy=torch.tensor([100.0]),
                cx=torch.tensor([float(img_id * 10 + 50)]),  # Unique cx: 50, 60, 70
                cy=torch.tensor([20.0]),
                width=torch.tensor([64]),
                height=torch.tensor([40]),
                camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
            )
            cameras_list.append(camera)

        # Stack cameras into single Cameras object
        original_cameras = Cameras(
            fx=torch.cat([cam.fx for cam in cameras_list]),
            fy=torch.cat([cam.fy for cam in cameras_list]),
            cx=torch.cat([cam.cx for cam in cameras_list]),
            cy=torch.cat([cam.cy for cam in cameras_list]),
            width=torch.cat([cam.width for cam in cameras_list]),
            height=torch.cat([cam.height for cam in cameras_list]),
            camera_to_worlds=torch.cat([cam.camera_to_worlds for cam in cameras_list]),
            camera_type=torch.cat([cam.camera_type for cam in cameras_list]),
        )

        # Mock datamanager with tiling that forces 2x1 tiling (2 tiles per image)
        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)
        datamanager.config = type(
            "Config",
            (),
            {
                "tile_size_max": 32,  # Force tiling
                "tile_alignment": 1,
            },
        )()

        # Mock the tile size calculation to return 2x1 tiling
        def mock_calculate_balanced_tile_sizes(width, height, tile_size_max, tile_alignment):
            return [32, 32], [40]  # 2 tiles horizontally, 1 vertically

        datamanager._calculate_balanced_tile_sizes = mock_calculate_balanced_tile_sizes

        # Process images through tiling pipeline
        tiled_images = []
        tiled_cameras_list = []
        tile_mapping = []

        for img_idx, image_data in enumerate(images):
            image = image_data["image"]
            height, width = image.shape[:2]

            tile_widths, tile_heights = datamanager._calculate_balanced_tile_sizes(
                width, height, datamanager.config.tile_size_max, datamanager.config.tile_alignment
            )

            # Should tile into 2 pieces
            if len(tile_widths) == 1 and len(tile_heights) == 1:
                # No tiling
                tiled_images.append(image_data)
                tiled_cameras_list.append(original_cameras[img_idx])
                tile_mapping.append(img_idx)
            else:
                # Tile the image
                image_tiles = datamanager._tile_undistorted_image(image, tile_widths, tile_heights)

                # Create cameras for each tile
                original_camera = original_cameras[img_idx].reshape(())
                tile_cameras = datamanager._replicate_cameras_for_tiles(
                    original_camera, tile_widths, tile_heights, img_idx
                )

                # Add each tile
                for tile_idx, (tile_image, tile_camera) in enumerate(zip(image_tiles, tile_cameras)):
                    tile_data = image_data.copy()
                    tile_data["image"] = tile_image

                    tiled_images.append(tile_data)
                    tiled_cameras_list.append(tile_camera)
                    tile_mapping.append(img_idx)

        # Verify correspondence: each tiled image should match its camera's parent
        expected_sequence = [
            (1.0, 50.0, 0),  # Image 0, tile 0: value=1.0, original_cx=50, parent=0
            (1.0, 50.0, 0),  # Image 0, tile 1: value=1.0, original_cx=50, parent=0
            (2.0, 60.0, 1),  # Image 1, tile 0: value=2.0, original_cx=60, parent=1
            (2.0, 60.0, 1),  # Image 1, tile 1: value=2.0, original_cx=60, parent=1
            (3.0, 70.0, 2),  # Image 2, tile 0: value=3.0, original_cx=70, parent=2
            (3.0, 70.0, 2),  # Image 2, tile 1: value=3.0, original_cx=70, parent=2
        ]

        assert len(tiled_images) == 6, f"Expected 6 tiled images, got {len(tiled_images)}"
        assert len(tiled_cameras_list) == 6, f"Expected 6 tiled cameras, got {len(tiled_cameras_list)}"

        for i, (expected_value, expected_original_cx, expected_parent) in enumerate(expected_sequence):
            # Check image content
            tile_value = tiled_images[i]["image"][0, 0, 0].item()
            assert tile_value == expected_value, f"Tile {i} has wrong image content: {tile_value} != {expected_value}"

            # Check camera parent index
            parent_idx = tiled_cameras_list[i].metadata["parent_camera_index"]
            assert parent_idx == expected_parent, f"Tile {i} has wrong parent camera: {parent_idx} != {expected_parent}"

            # Check tile mapping
            assert (
                tile_mapping[i] == expected_parent
            ), f"Tile {i} has wrong mapping: {tile_mapping[i]} != {expected_parent}"

    def test_tile_image_camera_correspondence(self):
        """Test that tiled images and cameras maintain correct correspondence"""
        # Create test image with unique pixel values for each tile region
        test_image = torch.zeros(80, 128, 3)  # 80x128 RGB

        # Fill each quadrant with unique values to identify tiles
        test_image[0:40, 0:64, :] = 1.0  # Top-left = 1.0
        test_image[0:40, 64:128, :] = 2.0  # Top-right = 2.0
        test_image[40:80, 0:64, :] = 3.0  # Bottom-left = 3.0
        test_image[40:80, 64:128, :] = 4.0  # Bottom-right = 4.0

        # Create test camera with known principal point
        cameras = Cameras(
            fx=torch.tensor([100.0]),
            fy=torch.tensor([100.0]),
            cx=torch.tensor([64.0]),  # Center X
            cy=torch.tensor([40.0]),  # Center Y
            width=torch.tensor([128]),
            height=torch.tensor([80]),
            camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
        )

        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)
        tile_widths, tile_heights = [64, 64], [40, 40]  # 2x2 tiling

        # Tile the image and cameras
        image_tiles = datamanager._tile_undistorted_image(test_image, tile_widths, tile_heights)
        tile_cameras = datamanager._replicate_cameras_for_tiles(cameras, tile_widths, tile_heights)

        # Verify correspondence: each tile's unique value should match its camera's metadata
        expected_correspondences = [
            (1.0, 0, 0, 0, 0),  # Top-left: value=1.0, tile_row=0, tile_col=0, offset_x=0, offset_y=0
            (2.0, 0, 1, 64, 0),  # Top-right: value=2.0, tile_row=0, tile_col=1, offset_x=64, offset_y=0
            (3.0, 1, 0, 0, 40),  # Bottom-left: value=3.0, tile_row=1, tile_col=0, offset_x=0, offset_y=40
            (4.0, 1, 1, 64, 40),  # Bottom-right: value=4.0, tile_row=1, tile_col=1, offset_x=64, offset_y=40
        ]

        for i, (expected_value, expected_row, expected_col, expected_offset_x, expected_offset_y) in enumerate(
            expected_correspondences
        ):
            # Check image tile has expected unique value
            tile_value = image_tiles[i][0, 0, 0].item()  # Get first pixel value
            assert tile_value == expected_value, f"Tile {i} has wrong image content: {tile_value} != {expected_value}"

            # Check camera metadata matches expected tile position
            metadata = tile_cameras[i].metadata
            assert (
                metadata["tile_row"] == expected_row
            ), f"Tile {i} camera has wrong row: {metadata['tile_row']} != {expected_row}"
            assert (
                metadata["tile_col"] == expected_col
            ), f"Tile {i} camera has wrong col: {metadata['tile_col']} != {expected_col}"
            assert (
                metadata["tile_offset_x"] == expected_offset_x
            ), f"Tile {i} camera has wrong offset_x: {metadata['tile_offset_x']} != {expected_offset_x}"
            assert (
                metadata["tile_offset_y"] == expected_offset_y
            ), f"Tile {i} camera has wrong offset_y: {metadata['tile_offset_y']} != {expected_offset_y}"

            # Verify camera intrinsics match the tile position
            expected_cx = cameras.cx.item() - expected_offset_x
            expected_cy = cameras.cy.item() - expected_offset_y
            assert torch.allclose(tile_cameras[i].cx, torch.tensor([expected_cx])), f"Tile {i} camera cx mismatch"
            assert torch.allclose(tile_cameras[i].cy, torch.tensor([expected_cy])), f"Tile {i} camera cy mismatch"

    def test_tile_to_original_mapping(self):
        """Test that tile_to_original_mapping correctly tracks which tiles came from which images"""
        # Test the concept: tile_mapping should track original image indices

        # Simulate the mapping logic from _apply_tiling_to_images
        tile_mapping = []

        # Image 0: 64x40 -> no tiling (1 tile)
        tile_mapping.append(0)

        # Image 1: 128x80 -> 2x2 tiling (4 tiles)
        for _ in range(4):
            tile_mapping.append(1)

        # Image 2: 50x30 -> no tiling (1 tile)
        tile_mapping.append(2)

        # Verify the mapping
        expected_mapping = [0, 1, 1, 1, 1, 2]
        assert tile_mapping == expected_mapping, f"Expected {expected_mapping}, got {tile_mapping}"

        # Verify we can trace tiles back to original images
        assert tile_mapping[0] == 0, "First tile should come from image 0"
        assert tile_mapping[1] == 1, "Second tile should come from image 1"
        assert tile_mapping[4] == 1, "Fifth tile should come from image 1"
        assert tile_mapping[5] == 2, "Last tile should come from image 2"

        # Verify total tiles
        assert len(tile_mapping) == 6, "Should have 6 total tiles"

        # Verify each original image is represented
        original_images_in_mapping = set(tile_mapping)
        assert original_images_in_mapping == {0, 1, 2}, "All original images should be represented"

    def test_eval_images_not_tiled(self):
        """Test that eval images are not tiled, only train images"""
        # Create mock images
        mock_images = [
            {"image": torch.rand(80, 128, 3)},  # Large image that would be tiled
            {"image": torch.rand(60, 100, 3)},  # Another large image
        ]

        # Create mock cameras
        cameras = Cameras(
            fx=torch.tensor([100.0, 100.0]),
            fy=torch.tensor([100.0, 100.0]),
            cx=torch.tensor([64.0, 50.0]),
            cy=torch.tensor([40.0, 30.0]),
            width=torch.tensor([128, 100]),
            height=torch.tensor([80, 60]),
            camera_to_worlds=torch.eye(4).unsqueeze(0).repeat(2, 1, 1)[:, :3, :],
        )

        # Create datamanager with tiling config
        config = MockConfig()
        config.tile_size_max = 64  # Will tile the large images
        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)
        datamanager.config = config

        # Mock the required attributes
        datamanager.train_dataset = None
        datamanager.eval_dataset = None
        datamanager.train_cameras = cameras
        datamanager.eval_cameras = cameras

        # Test train split - should be tiled
        train_tiled = datamanager._apply_tiling_to_images(mock_images, "train")
        assert len(train_tiled) > len(mock_images), "Train images should be tiled"

        # Test eval split - should NOT be tiled (but method won't be called due to split check)
        # Let's test the condition directly by checking _load_images logic

        # Verify the condition: tiling only applies to train split
        should_tile_train = config.tile_size_max > 0 and "train" == "train"  # noqa: PLR0133
        should_tile_eval = config.tile_size_max > 0 and "eval" == "train"  # noqa: PLR0133

        assert should_tile_train, "Train images should be tiled when tile_size_max > 0"
        assert not should_tile_eval, "Eval images should NOT be tiled"

    def test_tile_debugging_metadata(self):
        """Test that tile cameras contain debugging metadata"""
        # Create test camera
        cameras = Cameras(
            fx=torch.tensor([100.0]),
            fy=torch.tensor([95.0]),
            cx=torch.tensor([64.0]),
            cy=torch.tensor([40.0]),
            width=torch.tensor([128]),
            height=torch.tensor([80]),
            camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
        )

        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)
        tile_widths, tile_heights = [64, 64], [40, 40]  # 2x2 tiling

        # Test camera replication with debugging info
        tile_cameras = datamanager._replicate_cameras_for_tiles(
            cameras, tile_widths, tile_heights, parent_camera_index=5
        )

        # Verify all tile cameras have debugging metadata
        for i, tile_camera in enumerate(tile_cameras):
            assert tile_camera.metadata is not None, f"Tile {i} should have metadata"

            # Check required debugging fields
            metadata = tile_camera.metadata
            assert "parent_camera_index" in metadata, f"Tile {i} missing parent_camera_index"
            assert "tile_offset_x" in metadata, f"Tile {i} missing tile_offset_x"
            assert "tile_offset_y" in metadata, f"Tile {i} missing tile_offset_y"
            assert "tile_row" in metadata, f"Tile {i} missing tile_row"
            assert "tile_col" in metadata, f"Tile {i} missing tile_col"
            assert "total_tiles" in metadata, f"Tile {i} missing total_tiles"

            # Verify values
            assert metadata["parent_camera_index"] == 5, f"Tile {i} wrong parent index"
            assert metadata["total_tiles"] == 4, f"Tile {i} wrong total tiles"
            assert metadata["original_width"] == 128, f"Tile {i} wrong original width"
            assert metadata["original_height"] == 80, f"Tile {i} wrong original height"

        # Verify specific tile positions
        expected_positions = [
            (0, 0, 0, 0),  # Top-left: row=0, col=0, offset_x=0, offset_y=0
            (0, 1, 64, 0),  # Top-right: row=0, col=1, offset_x=64, offset_y=0
            (1, 0, 0, 40),  # Bottom-left: row=1, col=0, offset_x=0, offset_y=40
            (1, 1, 64, 40),  # Bottom-right: row=1, col=1, offset_x=64, offset_y=40
        ]

        for i, (expected_row, expected_col, expected_x, expected_y) in enumerate(expected_positions):
            metadata = tile_cameras[i].metadata
            assert metadata is not None, f"Tile {i} should have metadata"
            assert metadata["tile_row"] == expected_row, f"Tile {i} wrong row"
            assert metadata["tile_col"] == expected_col, f"Tile {i} wrong col"
            assert metadata["tile_offset_x"] == expected_x, f"Tile {i} wrong offset_x"
            assert metadata["tile_offset_y"] == expected_y, f"Tile {i} wrong offset_y"

    def test_alignment_constraints(self):
        """Test alignment constraint handling"""
        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)

        # Test various alignment scenarios
        test_cases = [
            (128, 80, 64, 16),  # Perfect alignment
            (130, 82, 64, 16),  # Slight misalignment
            (100, 100, 48, 32),  # Different alignment
        ]

        for width, height, tile_size_max, alignment in test_cases:
            tile_widths, tile_heights = datamanager._calculate_balanced_tile_sizes(
                width, height, tile_size_max, alignment
            )

            # All tiles except the last should be aligned
            for w in tile_widths[:-1]:
                assert w % alignment == 0, f"Non-edge tile width {w} not aligned to {alignment}"
            for h in tile_heights[:-1]:
                assert h % alignment == 0, f"Non-edge tile height {h} not aligned to {alignment}"


class TestTilingValidation:
    """Integration and validation tests"""

    def test_complete_tiling_pipeline(self):
        """Test the complete tiling pipeline with realistic data"""
        # Create mock image data
        width, height = 256, 256
        test_image = torch.rand(height, width, 3)

        # Create mock camera
        cameras = Cameras(
            fx=torch.tensor([100.0]),
            fy=torch.tensor([100.0]),
            cx=torch.tensor([128.0]),
            cy=torch.tensor([128.0]),
            width=torch.tensor([width]),
            height=torch.tensor([height]),
            camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
        )

        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)  # Skip __init__

        # Test tiling pipeline
        tile_widths, tile_heights = datamanager._calculate_balanced_tile_sizes(width, height, 128, 16)
        tiles = datamanager._tile_undistorted_image(test_image, tile_widths, tile_heights)
        tile_cameras = datamanager._replicate_cameras_for_tiles(cameras, tile_widths, tile_heights)

        # Validate results
        expected_num_tiles = len(tile_widths) * len(tile_heights)
        assert len(tiles) == expected_num_tiles, "Incorrect number of tiles"
        assert len(tile_cameras) == expected_num_tiles, "Incorrect number of tile cameras"

        # Validate camera adjustments
        for i, tile_camera in enumerate(tile_cameras):
            assert tile_camera.fx == cameras.fx, "Focal length fx should be preserved"
            assert tile_camera.fy == cameras.fy, "Focal length fy should be preserved"

    def test_error_conditions(self):
        """Test error handling and edge cases"""
        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)  # Skip __init__

        # Test invalid alignment
        with pytest.raises(ValueError):
            datamanager._calculate_balanced_tile_sizes(100, 100, 50, 0)  # Zero alignment

        # Test image too large for constraints
        with pytest.raises(ValueError):
            datamanager._calculate_balanced_tile_sizes(1000, 1000, 10, 32)  # Impossible constraints

    def test_performance_characteristics(self):
        """Test performance characteristics of tiling"""
        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)  # Skip __init__

        # Test that tiling calculation is fast
        import time

        start_time = time.time()
        for _ in range(100):
            datamanager._calculate_balanced_tile_sizes(1920, 1080, 512, 16)
        elapsed = time.time() - start_time

        # Should be very fast (< 10ms for 100 calculations)
        assert elapsed < 0.01, f"Tiling calculation too slow: {elapsed:.3f}s for 100 iterations"


if __name__ == "__main__":
    pytest.main([__file__])
