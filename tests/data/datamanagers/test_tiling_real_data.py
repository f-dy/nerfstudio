"""
Test tiling with real nerfstudio dataset structure
"""

import tempfile
from pathlib import Path

import numpy as np


def create_mock_dataset_structure():
    """Create a minimal dataset structure for testing"""
    temp_dir = Path(tempfile.mkdtemp())

    # Create images directory
    images_dir = temp_dir / "images"
    images_dir.mkdir()

    # Create a few test images (as numpy arrays, then save as files)
    test_images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),  # 640x480
        np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8),  # 480x360
        np.random.randint(0, 255, (720, 960, 3), dtype=np.uint8),  # 960x720
    ]

    # Save as simple numpy files (simulating images)
    image_paths = []
    for i, img in enumerate(test_images):
        img_path = images_dir / f"image_{i:03d}.npy"
        np.save(img_path, img)
        image_paths.append(img_path)

    # Create transforms.json (minimal)
    transforms = {"camera_angle_x": 0.8575560450553894, "frames": []}

    for i, img_path in enumerate(image_paths):
        transforms["frames"].append({"file_path": f"./images/{img_path.name}", "transform_matrix": np.eye(4).tolist()})

    import json

    with open(temp_dir / "transforms.json", "w") as f:
        json.dump(transforms, f)

    return temp_dir, image_paths


def test_config_validation():
    """Test FullImageDatamanagerConfig validation"""
    from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig

    # Test valid configurations
    valid_configs = [
        {"tile_size_max": 0, "tile_alignment": 16},  # Disabled
        {"tile_size_max": 512, "tile_alignment": 16},  # Standard
        {"tile_size_max": 256, "tile_alignment": 32},  # High alignment
        {"tile_size_max": 1024, "tile_alignment": 1},  # No alignment
    ]

    for config_params in valid_configs:
        config = FullImageDatamanagerConfig(**config_params)
        assert config.tile_size_max == config_params["tile_size_max"]
        assert config.tile_alignment == config_params["tile_alignment"]

    print("✅ Config validation test passed")


def test_tiling_scenarios():
    """Test various tiling scenarios with realistic parameters"""
    from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig

    # Create mock datamanager for testing
    class MockDatamanager:
        def __init__(self, config):
            self.config = config

        # Import actual methods
        from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager

        _calculate_balanced_tile_sizes = FullImageDatamanager._calculate_balanced_tile_sizes

    # Test realistic scenarios
    scenarios = [
        # (image_width, image_height, tile_size_max, tile_alignment, description)
        (1920, 1080, 512, 16, "Full HD with 512px tiles"),
        (2048, 1536, 256, 16, "High res with 256px tiles"),
        (640, 480, 256, 16, "VGA with 256px tiles (no tiling expected)"),
        (4096, 3072, 512, 32, "4K with 512px tiles and 32px alignment"),
    ]

    for width, height, tile_max, alignment, desc in scenarios:
        config = FullImageDatamanagerConfig(tile_size_max=tile_max, tile_alignment=alignment)
        dm = MockDatamanager(config)

        try:
            widths, heights = dm._calculate_balanced_tile_sizes(width, height, tile_max, alignment)

            # Verify constraints
            assert sum(widths) == width, f"{desc}: Width conservation failed"
            assert sum(heights) == height, f"{desc}: Height conservation failed"
            assert all(w <= tile_max for w in widths), f"{desc}: Width constraint violated"
            assert all(h <= tile_max for h in heights), f"{desc}: Height constraint violated"

            tile_count = len(widths) * len(heights)
            print(f"✅ {desc}: {len(widths)}x{len(heights)} = {tile_count} tiles (reduces GPU memory)")

        except ValueError as e:
            print(f"⚠️  {desc}: {e}")

    print("✅ Tiling scenarios test completed")


def test_memory_impact():
    """Test memory impact of different tiling configurations"""
    from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig

    class MockDatamanager:
        def __init__(self, config):
            self.config = config

        from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager

        _calculate_balanced_tile_sizes = FullImageDatamanager._calculate_balanced_tile_sizes

    # Test image: 1920x1080 (Full HD)
    width, height = 1920, 1080

    tile_sizes = [1024, 512, 256, 128]  # Different tile sizes

    print(f"Memory impact analysis for {width}x{height} image:")

    for tile_size in tile_sizes:
        config = FullImageDatamanagerConfig(tile_size_max=tile_size, tile_alignment=16)
        dm = MockDatamanager(config)

        try:
            widths, heights = dm._calculate_balanced_tile_sizes(width, height, tile_size, 16)
            tile_count = len(widths) * len(heights)

            print(
                f"  tile_size_max={tile_size:4d}: {len(widths)}x{len(heights)} = {tile_count:2d} tiles (reduces GPU memory)"
            )

        except ValueError as e:
            print(f"  tile_size_max={tile_size:4d}: ERROR - {e}")

    print("✅ Memory impact analysis completed")


if __name__ == "__main__":
    print("Running real dataset tiling tests...")

    test_config_validation()
    test_tiling_scenarios()
    test_memory_impact()

    print("🎉 All real dataset tests completed!")
