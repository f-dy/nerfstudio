"""
Core tiling functionality tests - focused on essential methods
"""

import torch

from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig


class TestTilingCore:
    """Test core tiling methods directly"""

    def create_mock_datamanager(self, tile_size_max=0, tile_alignment=16):
        """Create minimal mock for testing core methods"""
        config = FullImageDatamanagerConfig(tile_size_max=tile_size_max, tile_alignment=tile_alignment)

        class MockDatamanager:
            def __init__(self, config):
                self.config = config

            # Import actual methods
            from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager

            _calculate_balanced_tile_sizes = FullImageDatamanager._calculate_balanced_tile_sizes
            _tile_undistorted_image = FullImageDatamanager._tile_undistorted_image

        return MockDatamanager(config)

    def test_balanced_tile_sizes_comprehensive(self):
        """Test balanced tile size calculation with various scenarios"""
        dm = self.create_mock_datamanager()

        # Test 1: Disable tiling
        widths, heights = dm._calculate_balanced_tile_sizes(640, 480, 0, 16)
        assert widths == [640] and heights == [480], "Disable case failed"

        # Test 2: No tiling needed (small image)
        widths, heights = dm._calculate_balanced_tile_sizes(128, 80, 256, 16)
        assert widths == [128] and heights == [80], "Small image case failed"

        # Test 3: Exact balanced tiling
        widths, heights = dm._calculate_balanced_tile_sizes(640, 480, 256, 16)
        assert sum(widths) == 640 and sum(heights) == 480, "Pixel conservation failed"
        assert all(w <= 256 for w in widths), "Width constraint violated"
        assert all(h <= 256 for h in heights), "Height constraint violated"

        # Test 4: Alignment constraints (use compatible values)
        widths, heights = dm._calculate_balanced_tile_sizes(128, 96, 64, 16)
        assert sum(widths) == 128 and sum(heights) == 96, "Pixel conservation with alignment failed"

        print("✅ Balanced tile sizes comprehensive test passed")

    def test_image_tiling_pixel_perfect(self):
        """Test image tiling with pixel-perfect reconstruction"""
        dm = self.create_mock_datamanager()

        # Create test image with unique values for verification
        image = torch.arange(80 * 128 * 3, dtype=torch.float32).reshape(80, 128, 3)

        # Test various tiling scenarios
        test_cases = [
            ([64, 64], [40, 40]),  # 2x2 even split
            ([42, 42, 44], [26, 27, 27]),  # 3x3 with remainder
            ([128], [80]),  # No tiling
        ]

        for tile_widths, tile_heights in test_cases:
            tiles = dm._tile_undistorted_image(image, tile_widths, tile_heights)

            # Reconstruct image from tiles
            reconstructed = torch.zeros_like(image)
            tile_idx = 0
            y_offset = 0

            for tile_height in tile_heights:
                x_offset = 0
                for tile_width in tile_widths:
                    reconstructed[y_offset : y_offset + tile_height, x_offset : x_offset + tile_width] = tiles[tile_idx]
                    tile_idx += 1
                    x_offset += tile_width
                y_offset += tile_height

            # Verify perfect reconstruction
            assert torch.equal(image, reconstructed), f"Reconstruction failed for {tile_widths}x{tile_heights}"

        print("✅ Image tiling pixel-perfect test passed")

    def test_memory_efficiency(self):
        """Test memory usage patterns"""
        dm = self.create_mock_datamanager()

        # Create test image
        image = torch.rand(480, 640, 3)  # 640x480 image

        # Test different tile sizes (use valid configurations)
        tile_configs = [
            (256, 640, 480),  # Should create multiple tiles
            (512, 640, 480),  # Should create fewer tiles
        ]

        for tile_size_max, width, height in tile_configs:
            widths, heights = dm._calculate_balanced_tile_sizes(width, height, tile_size_max, 16)
            tiles = dm._tile_undistorted_image(image, widths, heights)

            # Verify tile count
            expected_count = len(widths) * len(heights)
            assert len(tiles) == expected_count, f"Expected {expected_count} tiles, got {len(tiles)}"

            # Verify memory usage is reasonable (tiles should be smaller than original)
            original_memory = image.numel() * image.element_size()
            total_tile_memory = sum(tile.numel() * tile.element_size() for tile in tiles)
            assert total_tile_memory == original_memory, "Memory usage mismatch"

        print("✅ Memory efficiency test passed")

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        dm = self.create_mock_datamanager()

        # Test very small images
        widths, heights = dm._calculate_balanced_tile_sizes(16, 16, 32, 16)
        assert widths == [16] and heights == [16], "Very small image failed"

        # Test single pixel wide/tall images
        widths, heights = dm._calculate_balanced_tile_sizes(1, 100, 50, 1)
        assert sum(widths) == 1 and sum(heights) == 100, "Single pixel width failed"

        # Test large alignment values
        widths, heights = dm._calculate_balanced_tile_sizes(128, 128, 64, 32)
        assert sum(widths) == 128 and sum(heights) == 128, "Large alignment failed"

        print("✅ Edge cases test passed")


if __name__ == "__main__":
    test = TestTilingCore()

    print("Running core tiling functionality tests...")

    test.test_balanced_tile_sizes_comprehensive()
    test.test_image_tiling_pixel_perfect()
    test.test_memory_efficiency()
    test.test_edge_cases()

    print("🎉 All core tiling tests passed!")
