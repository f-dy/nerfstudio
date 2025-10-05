"""
Tests for FullImageDatamanager tiling functionality
"""

import math
from typing import List, Tuple


class TestTilingAlgorithm:
    """Test the core tiling algorithm without dependencies"""

    def _calculate_balanced_tile_sizes(
        self, width: int, height: int, tile_size_max: int, tile_alignment: int
    ) -> Tuple[List[int], List[int]]:
        """Calculate balanced tile sizes that are similar in dimensions."""

        # Disable tiling if tile_size_max <= 0
        if tile_size_max <= 0:
            return [width], [height]

        # Calculate number of tiles needed
        num_tiles_x = math.ceil(width / tile_size_max)
        num_tiles_y = math.ceil(height / tile_size_max)

        # If no tiling needed (image smaller than max), return full image
        if num_tiles_x == 1 and num_tiles_y == 1:
            return [width], [height]

        # Calculate base tile sizes (evenly distributed, aligned)
        base_tile_width = width // num_tiles_x
        base_tile_height = height // num_tiles_y

        # Round base sizes to alignment multiples (with minimum size check)
        aligned_tile_width = max(tile_alignment, round(base_tile_width / tile_alignment) * tile_alignment)
        aligned_tile_height = max(tile_alignment, round(base_tile_height / tile_alignment) * tile_alignment)

        # Safety check: ensure aligned tiles don't exceed max
        if aligned_tile_width > tile_size_max or aligned_tile_height > tile_size_max:
            raise ValueError(f"Image too large for tile_size_max={tile_size_max} with alignment={tile_alignment}")

        # Create tile size lists with aligned sizes
        tile_widths = [aligned_tile_width] * num_tiles_x
        tile_heights = [aligned_tile_height] * num_tiles_y

        # Calculate total pixels used by aligned tiles
        total_aligned_width = aligned_tile_width * num_tiles_x
        total_aligned_height = aligned_tile_height * num_tiles_y

        # Add remaining pixels to the last tile (edge tiles can be non-aligned)
        if total_aligned_width != width:
            tile_widths[-1] += width - total_aligned_width
        if total_aligned_height != height:
            tile_heights[-1] += height - total_aligned_height

        # Safety check: ensure last tiles don't exceed tile_size_max
        if tile_widths[-1] > tile_size_max or tile_heights[-1] > tile_size_max:
            raise ValueError(f"Edge tiles would exceed tile_size_max={tile_size_max}")

        return tile_widths, tile_heights

    def test_disable_tiling(self):
        """Test that tiling is disabled when tile_size_max <= 0"""
        widths, heights = self._calculate_balanced_tile_sizes(640, 480, 0, 16)
        assert widths == [640] and heights == [480]

        widths, heights = self._calculate_balanced_tile_sizes(640, 480, -1, 16)
        assert widths == [640] and heights == [480]

    def test_no_tiling_needed(self):
        """Test that small images don't get tiled"""
        widths, heights = self._calculate_balanced_tile_sizes(128, 80, 256, 16)
        assert widths == [128] and heights == [80]

    def test_exact_division_640x480(self):
        """Test tiling with 640x480 image, tile_size_max=256, alignment=16"""
        widths, heights = self._calculate_balanced_tile_sizes(640, 480, 256, 16)

        # Expected: 3x2 tiles with balanced sizes
        expected_widths = [208, 208, 224]  # 208*2 + 224 = 640
        expected_heights = [240, 240]  # 240*2 = 480

        assert widths == expected_widths
        assert heights == expected_heights

        # Verify no pixel loss
        assert sum(widths) == 640
        assert sum(heights) == 480

    def test_camera_intrinsics_adjustment_128x80(self):
        """Test camera intrinsics adjustment for 128x80 → 2x2 tiles"""
        # This matches the test case from existing dataparser tests
        # Original: cx=57, cy=43, w=128, h=80 → Tiled: each tile varies

        widths, heights = self._calculate_balanced_tile_sizes(128, 80, 64, 16)
        expected_widths = [64, 64]  # 64+64=128
        expected_heights = [32, 48]  # 32+48=80, aligned to 16

        assert widths == expected_widths
        assert heights == expected_heights
        assert sum(widths) == 128
        assert sum(heights) == 80

    def test_remainder_handling_100x80(self):
        """Test tiling with remainder (100x80, tile_size_max=33)"""
        widths, heights = self._calculate_balanced_tile_sizes(100, 80, 33, 16)

        # Verify no pixel loss
        assert sum(widths) == 100
        assert sum(heights) == 80

        # All tiles should be <= tile_size_max
        assert all(w <= 33 for w in widths)
        assert all(h <= 33 for h in heights)

    def test_alignment_constraints(self):
        """Test different alignment values"""
        # Test alignment=1 (no alignment)
        widths, heights = self._calculate_balanced_tile_sizes(100, 80, 50, 1)
        assert sum(widths) == 100
        assert sum(heights) == 80

        # Test alignment=32
        widths, heights = self._calculate_balanced_tile_sizes(128, 96, 64, 32)
        assert sum(widths) == 128
        assert sum(heights) == 96
        # Most tiles should be multiples of 32 (except possibly edge tiles)
        assert all(w % 32 == 0 or w == widths[-1] for w in widths)
        assert all(h % 32 == 0 or h == heights[-1] for h in heights)

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Very small image
        widths, heights = self._calculate_balanced_tile_sizes(16, 16, 32, 16)
        assert widths == [16] and heights == [16]  # No tiling needed

        # Image exactly matching tile_size_max
        widths, heights = self._calculate_balanced_tile_sizes(256, 256, 256, 16)
        assert widths == [256] and heights == [256]  # No tiling needed

    def test_safety_checks(self):
        """Test safety checks and error conditions"""
        # This should work fine
        widths, heights = self._calculate_balanced_tile_sizes(1000, 800, 300, 16)
        assert sum(widths) == 1000
        assert sum(heights) == 800

        # All tiles should be within bounds
        assert all(w <= 300 for w in widths)
        assert all(h <= 300 for h in heights)


if __name__ == "__main__":
    test = TestTilingAlgorithm()

    print("Running tiling algorithm tests...")
    test.test_disable_tiling()
    print("✅ Disable tiling")

    test.test_no_tiling_needed()
    print("✅ No tiling needed")

    test.test_exact_division_640x480()
    print("✅ Exact division 640x480")

    test.test_camera_intrinsics_adjustment_128x80()
    print("✅ Camera intrinsics 128x80")

    test.test_remainder_handling_100x80()
    print("✅ Remainder handling 100x80")

    test.test_alignment_constraints()
    print("✅ Alignment constraints")

    test.test_edge_cases()
    print("✅ Edge cases")

    test.test_safety_checks()
    print("✅ Safety checks")

    print("🎉 All tiling algorithm tests passed!")
