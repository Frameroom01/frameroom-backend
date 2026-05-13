"""
Frameroom Backend — Phase 5
FastAPI server for image processing operations.
Handles lens correction, distortion, and perspective fixes.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import uvicorn

app = FastAPI(title="Frameroom Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def b64_to_cv2(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_b64(img: np.ndarray, quality: int = 92) -> str:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

def smart_fill_corners(img: np.ndarray, mask) -> np.ndarray:
    """Fill black corner areas from perspective warp using inpainting."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(grey, 5, 255, cv2.THRESH_BINARY_INV)
    h, w = img.shape[:2]
    center_mask = np.zeros((h, w), dtype=np.uint8)
    margin = int(min(h, w) * 0.15)
    center_mask[margin:h-margin, margin:w-margin] = 255
    fill_mask = cv2.bitwise_and(black_mask, cv2.bitwise_not(center_mask))
    if np.sum(fill_mask > 0) > (h * w * 0.001):
        kernel = np.ones((3, 3), np.uint8)
        fill_mask = cv2.dilate(fill_mask, kernel, iterations=1)
        return cv2.inpaint(img, fill_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    return img

def auto_crop_black_borders(img: np.ndarray, threshold: int = 8) -> np.ndarray:
    """Crop any thin black borders left after perspective correction."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    orig_h, orig_w = img.shape[:2]
    if x > orig_w * 0.01 or y > orig_h * 0.01:
        cropped = img[y:y+h, x:x+w]
        return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    return img


class LensCorrectionRequest(BaseModel):
    image: str
    distortion: float
    vertical: float
    horizontal: float

class AutoCorrectRequest(BaseModel):
    image: str


@app.get("/")
def root():
    return {"status": "Frameroom backend running", "version": "1.1.0"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/lens-correction")
async def lens_correction(req: LensCorrectionRequest):
    try:
        img = b64_to_cv2(req.image)
        h, w = img.shape[:2]

        # Downsample large images for processing speed
        # Perspective correction quality is identical at lower resolution
        MAX_DIM = 2400
        scale = 1.0
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        result = img.copy()

        if abs(req.distortion) > 0.5:
            k1 = -(req.distortion / 100.0) * 0.35
            k2 = k1 * 0.15
            cx, cy = w / 2.0, h / 2.0
            focal = max(w, h) * 0.85
            camera_matrix = np.array([[focal,0,cx],[0,focal,cy],[0,0,1]], dtype=np.float64)
            dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 0.85, (w, h))
            result = cv2.undistort(result, camera_matrix, dist_coeffs, None, new_camera_matrix)
            x, y, rw, rh = roi
            if rw > 0 and rh > 0:
                result = result[y:y+rh, x:x+rw]
                result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if abs(req.vertical) > 0.5 or abs(req.horizontal) > 0.5:
            h2, w2 = result.shape[:2]
            vert_factor  = req.vertical   / 100.0 * 0.25
            horiz_factor = req.horizontal / 100.0 * 0.25
            src = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]])
            v_shift = vert_factor  * w2 * 0.5
            h_shift = horiz_factor * h2 * 0.5
            dst = np.float32([
                [0   + v_shift + h_shift, 0 ],
                [w2  - v_shift + h_shift, 0 ],
                [w2  + v_shift - h_shift, h2],
                [0   - v_shift - h_shift, h2]
            ])
            M = cv2.getPerspectiveTransform(src, dst)

            # Step 1: Warp with black border
            result = cv2.warpPerspective(
                result, M, (w2, h2),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            # Step 2: Directly detect ALL black pixels created by the warp
            # and fill them — no prediction needed, catches everything
            grey_r = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, black_mask = cv2.threshold(grey_r, 10, 255, cv2.THRESH_BINARY_INV)

            # Exclude any legitimate dark content in the scene center
            # by only filling pixels that are connected to the image border
            # (warp artifacts always touch the border; real dark areas don't)
            border_connected = np.zeros_like(black_mask)
            # Flood fill from all 4 edges to find border-connected black areas
            h2, w2 = result.shape[:2]
            temp = black_mask.copy()
            # Seed from top, bottom, left, right edges
            for x in range(w2):
                if temp[0, x] > 0:
                    cv2.floodFill(temp, None, (x, 0), 128)
                if temp[h2-1, x] > 0:
                    cv2.floodFill(temp, None, (x, h2-1), 128)
            for y in range(h2):
                if temp[y, 0] > 0:
                    cv2.floodFill(temp, None, (0, y), 128)
                if temp[y, w2-1] > 0:
                    cv2.floodFill(temp, None, (w2-1, y), 128)
            # The filled regions (value=128) are the warp artifacts
            fill_mask = np.where(temp == 128, 255, 0).astype(np.uint8)

            # Dilate to catch edge pixels right at the boundary
            kernel = np.ones((5, 5), np.uint8)
            fill_mask = cv2.dilate(fill_mask, kernel, iterations=2)

            if np.sum(fill_mask > 0) > 10:
                correction_magnitude = abs(req.vertical) + abs(req.horizontal)
                fill_ratio = np.sum(fill_mask > 0) / (h2 * w2)

                if fill_ratio < 0.04:
                    # Small corners (<4% of image) — inpaint looks great
                    inpaint_radius = int(np.clip(correction_magnitude * 0.5, 6, 16))
                    result = cv2.inpaint(result, fill_mask, inpaintRadius=inpaint_radius,
                                         flags=cv2.INPAINT_TELEA)
                    # Second pass for any remnants
                    grey_r2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    _, rem = cv2.threshold(grey_r2, 10, 255, cv2.THRESH_BINARY_INV)
                    rem = cv2.bitwise_and(rem, fill_mask)
                    if np.sum(rem > 0) > 5:
                        result = cv2.inpaint(result, rem,
                                             inpaintRadius=inpaint_radius + 4,
                                             flags=cv2.INPAINT_TELEA)
                else:
                    # Large corners (>4% of image) — crop is cleaner than blurry inpaint
                    # Find the largest rectangle that contains no black pixels
                    # by cropping until all borders are non-black
                    grey_c = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    # Work inward from each edge until we hit non-black content
                    top = 0
                    while top < h2 // 3 and np.mean(grey_c[top, :]) < 15:
                        top += 1
                    bottom = h2 - 1
                    while bottom > h2 * 2 // 3 and np.mean(grey_c[bottom, :]) < 15:
                        bottom -= 1
                    left = 0
                    while left < w2 // 3 and np.mean(grey_c[:, left]) < 15:
                        left += 1
                    right = w2 - 1
                    while right > w2 * 2 // 3 and np.mean(grey_c[:, right]) < 15:
                        right -= 1

                    # Add a small buffer to ensure clean edges
                    buf = max(4, int(min(h2, w2) * 0.01))
                    top    = min(top    + buf, h2 // 3)
                    bottom = max(bottom - buf, h2 * 2 // 3)
                    left   = min(left   + buf, w2 // 3)
                    right  = max(right  - buf, w2 * 2 // 3)

                    if bottom > top and right > left:
                        cropped = result[top:bottom, left:right]
                        # Resize back to original dimensions with high quality
                        result = cv2.resize(cropped, (w2, h2),
                                            interpolation=cv2.INTER_LANCZOS4)

            # Step 3: Final crop of any thin remaining black lines
            result = auto_crop_black_borders(result, threshold=12)

        return {"success": True, "image": cv2_to_b64(result)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lens correction failed: {str(e)}")


@app.post("/auto-lens-correct")
async def auto_lens_correct(req: AutoCorrectRequest):
    """
    Multi-strategy detection:
    1. Hough line detection (interiors)
    2. Sobel edge analysis (exteriors/organic scenes — fallback)
    """
    try:
        img = b64_to_cv2(req.image)
        h, w = img.shape[:2]

        # Downsample for faster processing — line detection works fine at lower res
        MAX_DIM = 2400
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute edges first — used by both scene classification and line detection
        edges = cv2.Canny(grey, 25, 90, apertureSize=3)

        # ── Scene classification: interior vs exterior ────────
        # Interior shots have: less sky, more uniform mid-tone walls,
        # higher edge density from furniture/architecture
        # Exterior shots have: more sky (bright top), more variation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Sky detection — bright, low-saturation pixels in top 30%
        top_third = hsv[:h//3, :, :]
        brightness = top_third[:,:,2]
        saturation = top_third[:,:,1]
        sky_pixels = np.sum((brightness > 160) & (saturation < 80))
        sky_ratio  = sky_pixels / (w * h // 3)
        is_exterior = sky_ratio > 0.12  # >12% sky-like pixels = exterior

        # Edge density — interiors have more edges per pixel
        edge_density = np.sum(edges > 0) / (h * w)
        is_interior  = edge_density > 0.08 and not is_exterior

        # Set sensitivity based on scene type
        if is_interior:
            hough_mult = 400   # gentler — walls look dramatic but need less correction
            sobel_mult = 80
            max_correction = 45
        else:
            hough_mult = 900   # stronger — subtle convergence needs more push
            sobel_mult = 180
            max_correction = 70

        # ── Core detection: measure convergence by comparing
        # the spread of vertical features at top vs bottom ────
        #
        # TRUE VERTICAL RULE:
        # - Lines wider at top than bottom → converging (leaning in) → negative correction
        # - Lines wider at bottom than top → diverging (leaning out) → positive correction
        # - Equal width top and bottom → already straight → no correction
        #
        # Method: find all strong vertical edges, split into left/right halves,
        # measure their average X position in the top third vs bottom third.
        # The difference tells us exactly how much convergence exists.

        # Use Sobel to find strong vertical edges (works for both interior + exterior)
        sx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)

        # Strong vertical edges = gradient mostly vertical, strong magnitude
        threshold_mag = np.percentile(mag, 80)
        vert_edge = (np.abs(sy) > np.abs(sx) * 1.5) & (mag > threshold_mag)

        detected_vertical   = 0.0
        detected_distortion = 0.0
        lines_found         = 0
        strategy_used       = "convergence"

        if np.sum(vert_edge) > 200:
            ys_all, xs_all = np.where(vert_edge)

            # Split into left half and right half
            left_mask  = xs_all < w // 2
            right_mask = xs_all >= w // 2

            # Split each half into top third and bottom third
            top_band    = h // 3
            bottom_band = h * 2 // 3

            def band_mean_x(xs, ys, y_min, y_max):
                """Average X position of edge pixels in a horizontal band."""
                band = (ys >= y_min) & (ys < y_max)
                if np.sum(band) < 10:
                    return None
                return float(np.mean(xs[band]))

            # Left side: measure X position at top vs bottom
            lx_top    = band_mean_x(xs_all[left_mask],  ys_all[left_mask],  0,           top_band)
            lx_bottom = band_mean_x(xs_all[left_mask],  ys_all[left_mask],  bottom_band, h)

            # Right side: measure X position at top vs bottom
            rx_top    = band_mean_x(xs_all[right_mask], ys_all[right_mask], 0,           top_band)
            rx_bottom = band_mean_x(xs_all[right_mask], ys_all[right_mask], bottom_band, h)

            if all(v is not None for v in [lx_top, lx_bottom, rx_top, rx_bottom]):
                # Width at top = distance between right and left edge clusters at top
                width_top    = rx_top    - lx_top
                width_bottom = rx_bottom - lx_bottom

                # Convergence ratio: how much narrower is the top vs bottom
                # Positive = wider at bottom (top converges) → needs negative correction
                # Negative = wider at top (bottom converges) → needs positive correction
                width_diff  = width_bottom - width_top  # positive = top is narrower
                width_ratio = width_diff / w  # normalize by image width

                # Apply scene-specific scaling
                correction = width_ratio * max_correction * 2.5
                detected_vertical = float(np.clip(-correction, -max_correction, max_correction))
                strategy_used = f"convergence({'interior' if is_interior else 'exterior'})"
                lines_found = int(np.sum(vert_edge))

        # Hough lines for distortion detection only
        edges = cv2.Canny(grey, 25, 90, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=w*0.10, maxLineGap=60)
        if lines is not None:
            h_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = 90.0 if x2==x1 else abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                if angle < 22:
                    h_lines.append(line[0])
            if len(h_lines) >= 2:
                devs = [((y1+y2)/2)-(y1+(y2-y1)*0.5)
                        for x1,y1,x2,y2 in h_lines if abs(x2-x1) > w*0.15]
                if devs:
                    detected_distortion = float(np.clip(np.mean(devs)*3, -45, 45))

        # Re-encode the downsampled image for the correction call
        # This avoids timeout on full-resolution images
        downsampled_b64 = cv2_to_b64(img)

        correction_req = LensCorrectionRequest(
            image=downsampled_b64,
            distortion=detected_distortion,
            vertical=detected_vertical,
            horizontal=0.0
        )
        correction_result = await lens_correction(correction_req)

        return {
            "success":     True,
            "image":       correction_result["image"],
            "detected": {
                "distortion": round(detected_distortion, 1),
                "vertical":   round(detected_vertical,   1),
                "horizontal": 0.0,
            },
            "lines_found":   lines_found,
            "strategy_used": strategy_used,
            "scene_type":    "interior" if is_interior else "exterior",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto correction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
