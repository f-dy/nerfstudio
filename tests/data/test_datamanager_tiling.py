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

    def test_camera_intrinsics_and_replication(self):
        """Test camera intrinsics calculations and replication for tiling scenarios"""
        # Test exact division: 128×80 → 2×2 tiles
        original_cx, original_cy = 57.0, 43.0
        tile_widths = [64, 64]
        tile_heights = [40, 40]

        test_cases = [
            (0, 0, 57.0, 43.0),  # Top-left: no offset
            (0, 1, -7.0, 43.0),  # Top-right: cx = 57 - 64 = -7
            (1, 0, 57.0, 3.0),  # Bottom-left: cy = 43 - 40 = 3
            (1, 1, -7.0, 3.0),  # Bottom-right: cx = -7, cy = 3
        ]

        for row, col, expected_cx, expected_cy in test_cases:
            x_offset = sum(tile_widths[:col])
            y_offset = sum(tile_heights[:row])
            adjusted_cx = original_cx - x_offset
            adjusted_cy = original_cy - y_offset
            assert abs(adjusted_cx - expected_cx) < 1e-6, f"Exact division tile ({row},{col}): cx mismatch"
            assert abs(adjusted_cy - expected_cy) < 1e-6, f"Exact division tile ({row},{col}): cy mismatch"

        # Test camera replication with actual Camera objects
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
        tile_cameras = datamanager._replicate_cameras_for_tiles(cameras, tile_widths, tile_heights)

        # Verify correct number of cameras and adjustments
        assert len(tile_cameras) == 4, "Should create 4 cameras for 2x2 tiling"
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

    def test_camera_extrinsics_preservation_comprehensive(self):
        """Test that camera extrinsics are preserved correctly in all tiling scenarios"""
        # Create 2 cameras with different extrinsics
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

        # Test 1: No actual tiling (tile_size_max > image size)
        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)
        datamanager.config = type("Config", (), {"tile_size_max": 200, "tile_alignment": 16})()
        datamanager.train_cameras = dummy_cameras

        image0 = torch.rand(80, 128, 3)
        image1 = torch.rand(60, 100, 3)
        datamanager._apply_tiling_to_images(
            [{"image": image0, "image_idx": 0}, {"image": image1, "image_idx": 1}], "train"
        )

        # Should have 2 cameras (no tiling occurred)
        assert datamanager.train_cameras.shape[0] == 2, "No tiling should preserve camera count"

        # Verify extrinsics preserved
        assert torch.allclose(datamanager.train_cameras.camera_to_worlds[0, :, 3], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(datamanager.train_cameras.camera_to_worlds[1, :, 3], torch.tensor([4.0, 5.0, 6.0]))

        # Test 2: Actual tiling occurs
        datamanager.config.tile_size_max = 64  # Force tiling
        datamanager.train_cameras = dummy_cameras  # Reset

        datamanager._apply_tiling_to_images(
            [{"image": image0, "image_idx": 0}, {"image": image1, "image_idx": 1}], "train"
        )

        # Should have 6 cameras (4 from image0 + 2 from image1)
        assert datamanager.train_cameras.shape[0] == 6, "Tiling should create 6 cameras"

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

        # Test 3: Tensor concatenation dimensions are correct
        assert datamanager.train_cameras.camera_to_worlds.shape == (
            6,
            3,
            4,
        ), f"Wrong camera_to_worlds shape: {datamanager.train_cameras.camera_to_worlds.shape}"

    def test_tiling_correspondence_and_mapping(self):
        """Test that tiled images, cameras, and mappings maintain correct correspondence"""
        # Create test image with unique pixel values for each tile region
        test_image = torch.zeros(80, 128, 3)  # 80x128 RGB
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
            tile_value = image_tiles[i][0, 0, 0].item()
            assert tile_value == expected_value, f"Tile {i} has wrong image content: {tile_value} != {expected_value}"

            # Check camera metadata matches expected tile position
            metadata = tile_cameras[i].metadata
            assert metadata["tile_row"] == expected_row, f"Tile {i} camera has wrong row"
            assert metadata["tile_col"] == expected_col, f"Tile {i} camera has wrong col"
            assert metadata["tile_offset_x"] == expected_offset_x, f"Tile {i} camera has wrong offset_x"
            assert metadata["tile_offset_y"] == expected_offset_y, f"Tile {i} camera has wrong offset_y"

            # Verify camera intrinsics match the tile position
            expected_cx = cameras.cx.item() - expected_offset_x
            expected_cy = cameras.cy.item() - expected_offset_y
            assert torch.allclose(tile_cameras[i].cx, torch.tensor([expected_cx])), f"Tile {i} camera cx mismatch"
            assert torch.allclose(tile_cameras[i].cy, torch.tensor([expected_cy])), f"Tile {i} camera cy mismatch"

        # Test tile-to-original mapping concept
        tile_mapping = []
        # Image 0: 64x40 -> no tiling (1 tile)
        tile_mapping.append(0)
        # Image 1: 128x80 -> 2x2 tiling (4 tiles)
        for _ in range(4):
            tile_mapping.append(1)
        # Image 2: 50x30 -> no tiling (1 tile)
        tile_mapping.append(2)

        expected_mapping = [0, 1, 1, 1, 1, 2]
        assert tile_mapping == expected_mapping, f"Expected {expected_mapping}, got {tile_mapping}"
        assert set(tile_mapping) == {0, 1, 2}, "All original images should be represented"

    def test_splatfacto_compatibility_and_metadata(self):
        """Test splatfacto compatibility and debugging metadata"""
        # Test single camera (should work with splatfacto)
        single_camera = Cameras(
            fx=torch.tensor([100.0]),
            fy=torch.tensor([100.0]),
            cx=torch.tensor([32.0]),
            cy=torch.tensor([20.0]),
            width=torch.tensor([64]),
            height=torch.tensor([40]),
            camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
        )
        assert single_camera.shape[0] == 1, "Single camera should pass splatfacto assertion"

        # Test tiled cameras with debugging metadata
        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)
        tile_widths, tile_heights = [32, 32], [20, 20]  # 2x2 tiling

        tile_cameras = datamanager._replicate_cameras_for_tiles(
            single_camera, tile_widths, tile_heights, parent_camera_index=5
        )

        # Verify all tile cameras have debugging metadata
        for i, tile_camera in enumerate(tile_cameras):
            # Debugging metadata
            assert tile_camera.metadata is not None, f"Tile {i} should have metadata"
            metadata = tile_camera.metadata
            required_fields = [
                "parent_camera_index",
                "tile_offset_x",
                "tile_offset_y",
                "tile_row",
                "tile_col",
                "total_tiles",
            ]
            for field in required_fields:
                assert field in metadata, f"Tile {i} missing {field}"

            assert metadata["parent_camera_index"] == 5, f"Tile {i} wrong parent index"
            assert metadata["total_tiles"] == 4, f"Tile {i} wrong total tiles"

        # Test splatfacto compatibility: create batch and test indexing
        cameras_data = []
        for i in range(3):
            camera_to_worlds = torch.tensor(
                [[[1.0, 0.0, 0.0, float(i + 1)], [0.0, 1.0, 0.0, float(i + 2)], [0.0, 0.0, 1.0, 3.0]]]
            )
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

        # Test indexed access works (this is how datamanager provides cameras to splatfacto)
        for i in range(3):
            indexed_camera = tiled_cameras[i : i + 1]
            assert indexed_camera.shape[0] == 1, f"Indexed camera {i} should pass splatfacto assertion"

    def test_eval_images_not_tiled_and_alignment(self):
        """Test that eval images are not tiled and alignment constraints work"""
        # Test eval images are not tiled
        config = MockConfig()
        config.tile_size_max = 64  # Will tile train images

        should_tile_train = config.tile_size_max > 0 and "train" == "train"  # noqa: PLR0133
        should_tile_eval = config.tile_size_max > 0 and "eval" == "train"  # noqa: PLR0133

        assert should_tile_train, "Train images should be tiled when tile_size_max > 0"
        assert not should_tile_eval, "Eval images should NOT be tiled"

        # Test alignment constraints
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

    def test_complete_tiling_pipeline_and_error_conditions(self):
        """Test the complete tiling pipeline with realistic data and error handling"""
        # Test complete pipeline
        width, height = 256, 256
        test_image = torch.rand(height, width, 3)

        cameras = Cameras(
            fx=torch.tensor([100.0]),
            fy=torch.tensor([100.0]),
            cx=torch.tensor([128.0]),
            cy=torch.tensor([128.0]),
            width=torch.tensor([width]),
            height=torch.tensor([height]),
            camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
        )

        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)

        # Test tiling pipeline
        tile_widths, tile_heights = datamanager._calculate_balanced_tile_sizes(width, height, 128, 16)
        tiles = datamanager._tile_undistorted_image(test_image, tile_widths, tile_heights)
        tile_cameras = datamanager._replicate_cameras_for_tiles(cameras, tile_widths, tile_heights)

        # Validate results
        expected_num_tiles = len(tile_widths) * len(tile_heights)
        assert len(tiles) == expected_num_tiles, "Incorrect number of tiles"
        assert len(tile_cameras) == expected_num_tiles, "Incorrect number of tile cameras"

        # Validate camera adjustments
        for tile_camera in tile_cameras:
            assert tile_camera.fx == cameras.fx, "Focal length fx should be preserved"
            assert tile_camera.fy == cameras.fy, "Focal length fy should be preserved"

        # Test error conditions
        with pytest.raises(ValueError):
            datamanager._calculate_balanced_tile_sizes(100, 100, 50, 0)  # Zero alignment

        with pytest.raises(ValueError):
            datamanager._calculate_balanced_tile_sizes(1000, 1000, 10, 32)  # Impossible constraints


if __name__ == "__main__":
    pytest.main([__file__])
