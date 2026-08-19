from spd_hot.georef import clip_xyxy_to_image


def test_clip_xyxy_to_image_bounds():
    x1, y1, x2, y2 = clip_xyxy_to_image(-10, -5, 280, 300, width=256, height=256)
    assert (x1, y1) == (0.0, 0.0)
    assert (x2, y2) == (256.0, 256.0)

