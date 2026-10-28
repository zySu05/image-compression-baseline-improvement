import math


class RateController:
    def __init__(self, model, max_steps=8):
        self.model = model
        self.max_steps = max_steps

    def compress_to_target(self, x, target_bpp):
        if not math.isfinite(target_bpp) or target_bpp <= 0:
            raise ValueError("target_bpp必须为有限正数")

        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError("当前仅支持单张图像，形状为[1,C,H,W]")

        pixels = x.shape[-2] * x.shape[-1]

        def encode(position):
            encoded = self.model.compress_at(x, position)
            bits = sum(
                len(stream) * 8
                for group in encoded["strings"]
                for stream in group
            )
            return bits / pixels, encoded

        left = 0.0
        right = float(self.model.log_gain.shape[0] - 1)

        low_bpp, low_data = encode(left)
        high_bpp, high_data = encode(right)

        if low_bpp > high_bpp:
            raise ValueError("码率端点顺序异常")

        if not low_bpp <= target_bpp <= high_bpp:
            raise ValueError(
                f"目标{target_bpp:.5f}超出当前图像可达范围"
                f"[{low_bpp:.5f}, {high_bpp:.5f}]"
            )

        best = min(
            [(left, low_bpp, low_data),
             (right, high_bpp, high_data)],
            key=lambda t: abs(t[1] - target_bpp)
        )

        for _ in range(self.max_steps):
            middle = (left + right) / 2
            actual_bpp, encoded = encode(middle)

            if abs(actual_bpp - target_bpp) < abs(
                best[1] - target_bpp
            ):
                best = (middle, actual_bpp, encoded)

            if actual_bpp < target_bpp:
                left = middle
            else:
                right = middle

        position, actual_bpp, encoded = best

        return {
            "position": position,
            "target_bpp": target_bpp,
            "actual_bpp": actual_bpp,
            "error": abs(actual_bpp - target_bpp),
            "encoded": encoded,
            "search_encodes": self.max_steps + 2,
        }
