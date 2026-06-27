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

                # width_bottom - width_top:
                #   POSITIVE = bottom wider = lines converge at top = need NEGATIVE correction
                #   NEGATIVE = top wider = lines diverge = need POSITIVE correction
                # The warp transform: positive v_shift narrows top, negative widens top
                # So to fix converging verticals (bottom wider): send NEGATIVE value
                width_diff  = width_bottom - width_top
                width_ratio = width_diff / w
                # positive width_diff → negative correction (widens the top)
                correction = -(width_ratio * max_correction * 2.5)
                detected_vertical = float(np.clip(correction, -max_correction, max_correction))

                strategy_used = f"conv-{'int' if is_interior else 'ext'}"
                lines_found = int(np.sum(vert_edge))

                # Include debug in response for diagnosis
                strategy_used += f" wt={width_top:.0f} wb={width_bottom:.0f} lxt={lx_top:.0f} lxb={lx_bottom:.0f} rxt={rx_top:.0f} rxb={rx_bottom:.0f}"
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
            "debug": strategy_used,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto correction failed: {str(e)}")




# ── WINDOW GLARE REMOVAL v2 ───────────────────────────────────
class WindowGlareRequest(BaseModel):
    image: str
    strength: float = 80.0
    recover_detail: bool = True


@app.post("/window-glare")
async def window_glare(req: WindowGlareRequest):
    """
    Multi-surface glare removal v2.
    Detects and reduces glare anywhere in the photo:
    - Blown-out windows
    - Table/floor/furniture specular highlights  
    - Wall reflections from lamps
    - Any overexposed region
    Two-pass: dim blowout → recover tonal detail from adjacent pixels.
    """
    try:
        img = b64_to_cv2(req.image)
        h, w = img.shape[:2]

        # Downsample for processing speed
        MAX_DIM = 2000
        scale = 1.0
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)),
                             interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        strength = req.strength / 100.0
        result_f = img.astype(np.float32)
        b_ch, g_ch, r_ch = cv2.split(img)

        # ── PASS 1: Detect all glare regions ─────────────────
        # Tier 1: Full blowout (all channels > 230)
        tier1 = (
            (r_ch.astype(np.int32) > 230) &
            (g_ch.astype(np.int32) > 230) &
            (b_ch.astype(np.int32) > 230)
        ).astype(np.uint8) * 255

        # Tier 2: Strong glare (all channels > 195)
        tier2 = (
            (r_ch.astype(np.int32) > 195) &
            (g_ch.astype(np.int32) > 195) &
            (b_ch.astype(np.int32) > 195)
        ).astype(np.uint8) * 255

        # Tier 3: Partial glare / specular highlights (luminance > 175)
        lum = (0.299*r_ch + 0.587*g_ch + 0.114*b_ch).astype(np.float32)
        tier3 = (lum > 175).astype(np.uint8) * 255

        # ── Ceiling light spots: small but very bright ────────
        # Recessed lights and spotlights — don't filter by size
        ceiling_spots = (
            (r_ch.astype(np.int32) > 245) &
            (g_ch.astype(np.int32) > 245) &
            (b_ch.astype(np.int32) > 245)
        ).astype(np.uint8) * 255
        # Include full ceiling area (top 40%)
        ceiling_zone = np.zeros((h, w), dtype=np.uint8)
        ceiling_zone[:int(h*0.4), :] = 255
        ceiling_spots = cv2.bitwise_and(ceiling_spots, ceiling_zone)
        k_ceil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
        ceiling_spots = cv2.morphologyEx(ceiling_spots, cv2.MORPH_CLOSE, k_ceil)

        # Connect nearby glare regions
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))
        tier1_closed = cv2.morphologyEx(tier1, cv2.MORPH_CLOSE, k_close)
        tier2_closed = cv2.morphologyEx(tier2, cv2.MORPH_CLOSE, k_close)
        tier3_closed = cv2.morphologyEx(tier3, cv2.MORPH_CLOSE,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8)))

        # ── Filter minimum size (keep ceiling lights, filter tiny specks) ─
        min_area = max(30, int(h * w * 0.0002))  # lower threshold

        def filter_small(mask, min_px):
            n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            out = np.zeros_like(mask)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] >= min_px:
                    out[lbl == i] = 255
            return out

        tier1_clean = filter_small(tier1_closed, min_area)
        tier2_clean = filter_small(tier2_closed, min_area)
        tier3_clean = filter_small(tier3_closed, min_area)
        # Ceiling spots: keep all sizes — recessed lights can be small
        ceiling_clean = ceiling_spots

        # ── Build soft masks with feathered edges ─────────────
        def soft_mask(mask, blur_r=25):
            return cv2.GaussianBlur(mask.astype(np.float32), (blur_r, blur_r), 0) / 255.0

        # Tighter feathering — smaller blur radius keeps correction closer to glare source
        mask1 = soft_mask(tier1_clean, 15)   # full blowout — tight
        mask2 = soft_mask(tier2_clean, 11)   # strong glare — tight
        mask3 = soft_mask(tier3_clean, 7)    # partial — very tight
        mask_ceil = soft_mask(ceiling_clean, 21)  # ceiling — moderate feather

        # Edge-aware boundary: use Canny edges to find window/wall boundaries
        # Stop the mask from crossing strong edges (window frame into wall)
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 40, 120)
        edge_barrier = 1.0 - cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 0) / 255.0 * 0.7
        mask1 = mask1 * edge_barrier
        mask2 = mask2 * edge_barrier
        mask3 = mask3 * edge_barrier

        # Count regions
        n1, _, _, _ = cv2.connectedComponentsWithStats(tier1_clean, 8)
        n2, _, _, _ = cv2.connectedComponentsWithStats(tier2_clean, 8)
        nc, _, _, _ = cv2.connectedComponentsWithStats(ceiling_clean, 8)
        total_regions = max(0, (n1-1) + (n2-1) + (nc-1))
        coverage = float(np.sum(tier2_clean > 0)) / (h * w) * 100

        # ── PASS 2: Smooth highlight tone curve ──────────────
        # Simple, clean approach: apply an S-curve to pull down
        # overexposed highlights without touching midtones or shadows
        # Works on BGR directly — no color space conversion artifacts
        result_f = img.astype(np.float32)

        # Build smooth highlight reduction curve using LUT
        # Pixels below 180: untouched
        # Pixels 180-255: gradually pulled down based on strength
        lut = np.arange(256, dtype=np.float32)
        threshold = 180.0
        for i in range(256):
            if i > threshold:
                # How far into the highlight range (0=at threshold, 1=at 255)
                t = (i - threshold) / (255.0 - threshold)
                # Smooth ease-in curve — gentle at start, stronger near 255
                ease = t * t
                # Pull down by up to 35% of the highlight value at full strength
                reduction = ease * strength * 0.35 * i
                lut[i] = max(threshold * 0.85, i - reduction)

        # Apply LUT to each channel independently
        for c in range(3):
            ch = result_f[:, :, c]
            # Vectorized LUT lookup
            ch_int = np.clip(ch, 0, 255).astype(np.uint8)
            corrected = lut[ch_int]
            # Only apply in glare mask areas — walls untouched
            correction_mask = np.clip(
                mask1 * 0.90 + mask2 * 0.70 + mask_ceil * 0.55,
                0, 1
            )
            result_f[:, :, c] = ch * (1 - correction_mask) + corrected * correction_mask

        # Very subtle specular reduction separately (even lighter touch)
        for c in range(3):
            ch = result_f[:, :, c]
            spec_target = ch * (1 - 0.15 * strength)
            result_f[:, :, c] = ch * (1 - mask3 * 0.20) + spec_target * (mask3 * 0.20)

        result = np.clip(result_f, 0, 255).astype(np.uint8)

        return {
            "success":       True,
            "image":         cv2_to_b64(result),
            "regions_found": total_regions,
            "coverage_pct":  round(coverage, 1),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Glare removal failed: {str(e)}"
        )


# ── WINDOW GLARE REMOVAL ──────────────────────────────────────
class WindowGlareRequest(BaseModel):
    image: str
    strength: float = 80.0    # 0-100 how aggressively to reduce glare
    recover_detail: bool = True  # try to recover window detail vs just darken


@app.post("/window-glare")
async def window_glare(req: WindowGlareRequest):
    """
    Detect and reduce window glare in real estate photos.
    
    Strategy:
    1. Detect overexposed regions (near-white pixels) likely to be windows
    2. Find connected bright regions and classify as windows vs lights
    3. For each window region, sample surrounding wall color
    4. Blend down the overexposed area using the wall tone as reference
    5. Optionally add a natural window tone (cool blue-grey for daylight)
    """
    try:
        img = b64_to_cv2(req.image)
        h, w = img.shape[:2]

        # Downsample for processing
        MAX_DIM = 2000
        scale = 1.0
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)),
                           interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        result = img.copy().astype(np.float32)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        strength = req.strength / 100.0

        # ── Step 1: Find overexposed regions ──────────────────
        # Pixels where all channels are very bright = blown out
        b, g, r = cv2.split(img)
        overexposed = (
            (r.astype(np.int32) > 220) &
            (g.astype(np.int32) > 220) &
            (b.astype(np.int32) > 220)
        ).astype(np.uint8) * 255

        # ── Step 2: Find connected bright regions ─────────────
        # Use morphological ops to connect nearby overexposed pixels
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        overexposed_closed = cv2.morphologyEx(
            overexposed, cv2.MORPH_CLOSE, kernel_close
        )

        # Find connected components (individual window blobs)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            overexposed_closed, connectivity=8
        )

        # ── Step 3: Filter to keep only window-sized regions ──
        # Windows are: large enough, not at very top (ceiling lights),
        # roughly rectangular, not too wide (full wall brightness)
        min_area = (h * w) * 0.003   # at least 0.3% of image
        max_area = (h * w) * 0.35    # at most 35% of image
        window_mask = np.zeros((h, w), dtype=np.uint8)
        windows_found = 0

        for label_id in range(1, num_labels):
            area      = stats[label_id, cv2.CC_STAT_AREA]
            cx_stat   = stats[label_id, cv2.CC_STAT_LEFT]
            cy_stat   = stats[label_id, cv2.CC_STAT_TOP]
            cw_stat   = stats[label_id, cv2.CC_STAT_WIDTH]
            ch_stat   = stats[label_id, cv2.CC_STAT_HEIGHT]
            cy_center = centroids[label_id][1]

            if area < min_area or area > max_area:
                continue

            # Skip tiny bright spots (lights, reflections)
            if cw_stat < w * 0.03 or ch_stat < h * 0.03:
                continue

            # Skip regions at very top of image (ceiling fixtures)
            if cy_center < h * 0.05:
                continue

            # This looks like a window — add to mask
            window_mask[labels == label_id] = 255
            windows_found += 1

        if windows_found == 0:
            # No clear windows found — try with lower threshold
            overexposed_loose = (
                (r.astype(np.int32) > 200) &
                (g.astype(np.int32) > 200) &
                (b.astype(np.int32) > 200)
            ).astype(np.uint8) * 255
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
            window_mask = cv2.morphologyEx(overexposed_loose, cv2.MORPH_CLOSE, kernel2)

        # ── Step 4: Expand mask slightly to catch edge glow ───
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        window_mask_expanded = cv2.dilate(window_mask, kernel_dilate, iterations=2)

        # Soft edge — feather the mask so correction blends naturally
        window_mask_float = cv2.GaussianBlur(
            window_mask_expanded.astype(np.float32), (31, 31), 0
        ) / 255.0

        # ── Step 5: Sample surrounding wall color ─────────────
        # Look at pixels just outside the window mask for reference tone
        # This tells us what the wall looks like without the glare influence
        kernel_surround = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        surround_area = cv2.dilate(window_mask, kernel_surround, iterations=1)
        surround_only = cv2.bitwise_and(
            surround_area,
            cv2.bitwise_not(window_mask_expanded)
        )

        # Average color of surrounding wall area
        surround_pixels = img[surround_only > 0]
        if len(surround_pixels) > 50:
            wall_color = np.mean(surround_pixels, axis=0)  # BGR
        else:
            # Fallback: use overall image average excluding bright areas
            non_bright = img[overexposed == 0]
            wall_color = np.mean(non_bright, axis=0) if len(non_bright) > 0 \
                         else np.array([180.0, 180.0, 180.0])

        # ── Step 6: Build target window color ─────────────────
        # Natural window appearance: slightly cool, moderately bright
        # Mix wall color with a daylight window tone
        daylight_window = np.array([
            wall_color[0] * 0.6 + 140,  # B: push toward cool blue
            wall_color[1] * 0.6 + 130,  # G: neutral
            wall_color[2] * 0.5 + 110,  # R: slightly desaturated
        ], dtype=np.float32)
        # Clamp to valid range
        daylight_window = np.clip(daylight_window, 80, 210)

        # ── Step 7: Apply correction to overexposed areas ─────
        result_f = img.astype(np.float32)

        # Build per-channel correction
        for c in range(3):
            channel = result_f[:, :, c]
            target  = daylight_window[c]

            # In the window area: blend toward target based on strength and mask
            # The more overexposed a pixel, the more we correct it
            pixel_overexp = np.clip((channel - 200) / 55.0, 0, 1)
            correction_amount = window_mask_float * pixel_overexp * strength

            # Pull overexposed pixels toward the target window color
            corrected = channel + correction_amount * (target - channel)
            result_f[:, :, c] = corrected

        # ── Step 8: If recover_detail, add subtle texture ─────
        # Real windows have subtle frame/pane lines visible
        # Apply a gentle sharpening to the window area to hint at detail
        if req.recover_detail:
            kernel_sharp = np.array([
                [ 0, -0.3,  0],
                [-0.3, 2.2, -0.3],
                [ 0, -0.3,  0]
            ], dtype=np.float32)
            sharpened = cv2.filter2D(result_f, -1, kernel_sharp)
            # Only apply sharpening inside window area
            sharp_blend = window_mask_float[:, :, np.newaxis] * 0.3
            result_f = result_f * (1 - sharp_blend) + sharpened * sharp_blend

        result = np.clip(result_f, 0, 255).astype(np.uint8)

        return {
            "success":        True,
            "image":          cv2_to_b64(result),
            "windows_found":  windows_found,
            "coverage_pct":   round(float(np.sum(window_mask > 0)) / (h*w) * 100, 1),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Window glare removal failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# ── HDR BLEND ─────────────────────────────────────────────────
class HDRBlendRequest(BaseModel):
    image_interior: str   # base64 — exposed for interior
    image_exterior: str   # base64 — exposed for windows/exterior
    blend_strength: float = 85.0   # 0-100 how much exterior exposure to blend in
    feather: float = 40.0          # 0-100 softness of transition edge


@app.post("/hdr-blend")
async def hdr_blend(req: HDRBlendRequest):
    """
    Blend two exposures of the same scene for HDR window correction.
    
    Takes:
    - Interior exposure: correctly exposed interior, blown-out windows
    - Exterior exposure: correctly exposed windows, dark interior
    
    Strategy:
    1. Align both images (handle slight camera movement between shots)
    2. Build luminance map to detect which exposure is better per region
    3. Use blown-out detection to identify window areas needing exterior exposure
    4. Blend with smooth feathered transition at window edges
    5. Apply tone matching so exposures look natural together
    """
    try:
        img_int = b64_to_cv2(req.image_interior)
        img_ext = b64_to_cv2(req.image_exterior)

        # Downsample for processing
        MAX_DIM = 2000
        h, w = img_int.shape[:2]
        scale = 1.0
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img_int = cv2.resize(img_int, (int(w*scale), int(h*scale)),
                                 interpolation=cv2.INTER_AREA)
            img_ext = cv2.resize(img_ext, (int(w*scale), int(h*scale)),
                                 interpolation=cv2.INTER_AREA)
            h, w = img_int.shape[:2]

        strength = req.blend_strength / 100.0
        feather  = max(5, int(req.feather / 100.0 * min(h, w) * 0.15))

        # ── Step 1: Align images (ECC algorithm) ──────────────
        # Handles slight camera movement between bracket shots
        grey_int = cv2.cvtColor(img_int, cv2.COLOR_BGR2GRAY)
        grey_ext = cv2.cvtColor(img_ext, cv2.COLOR_BGR2GRAY)

        try:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            _, warp_matrix = cv2.findTransformECC(
                grey_int, grey_ext, warp_matrix,
                cv2.MOTION_TRANSLATION, criteria
            )
            img_ext_aligned = cv2.warpAffine(
                img_ext, warp_matrix, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
        except Exception:
            # If alignment fails, use as-is
            img_ext_aligned = img_ext

        # ── Step 2: Build blend mask ───────────────────────────
        # Detect overexposed regions in interior shot (windows)
        # These are the areas where we want exterior exposure
        b_i, g_i, r_i = cv2.split(img_int)

        # Primary mask: fully blown pixels
        blown = (
            (r_i.astype(np.int32) > 220) &
            (g_i.astype(np.int32) > 220) &
            (b_i.astype(np.int32) > 220)
        ).astype(np.uint8) * 255

        # Secondary mask: bright pixels that need balancing
        bright = (
            (r_i.astype(np.int32) > 190) &
            (g_i.astype(np.int32) > 190) &
            (b_i.astype(np.int32) > 190)
        ).astype(np.uint8) * 255

        # Connect nearby bright regions
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        blown_closed = cv2.morphologyEx(blown, cv2.MORPH_CLOSE, k)
        bright_closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k)

        # Filter small regions (not windows)
        min_area = int(h * w * 0.002)

        def keep_large(mask, min_px):
            n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            out = np.zeros_like(mask)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] >= min_px:
                    out[lbl == i] = 255
            return out

        blown_clean  = keep_large(blown_closed,  min_area)
        bright_clean = keep_large(bright_closed, min_area)

        # Combine: fully blown gets full exterior, bright gets partial
        blend_mask = np.zeros((h, w), dtype=np.float32)
        blend_mask[blown_clean  > 0] = 1.0
        blend_mask[bright_clean > 0] = np.maximum(
            blend_mask[bright_clean > 0], 0.6
        )

        # Feather edges for smooth transition
        feather_size = feather * 2 + 1
        blend_mask = cv2.GaussianBlur(blend_mask, (feather_size, feather_size), 0)
        blend_mask = np.clip(blend_mask * strength, 0, strength)

        # ── Step 3: Luminance check ────────────────────────────
        # Only blend exterior where it's actually better (darker = more detail)
        lum_int = cv2.cvtColor(img_int, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lum_ext = cv2.cvtColor(img_ext_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Where exterior is brighter than interior, don't blend
        # (exterior overexposed that region too — no benefit)
        ext_darker = (lum_ext < lum_int).astype(np.float32)
        ext_darker_smooth = cv2.GaussianBlur(ext_darker, (21, 21), 0)
        blend_mask = blend_mask * ext_darker_smooth

        # ── Step 4: Tone match exterior to interior ────────────
        # Match the color/brightness of non-window areas so they look
        # like they came from the same photo
        non_window = (blend_mask < 0.1).astype(np.uint8)
        result_f = img_int.astype(np.float32)

        for c in range(3):
            int_pixels = img_int[:, :, c][non_window > 0].astype(np.float32)
            ext_pixels = img_ext_aligned[:, :, c][non_window > 0].astype(np.float32)

            if len(int_pixels) > 100 and len(ext_pixels) > 100:
                int_mean = np.mean(int_pixels)
                ext_mean = np.mean(ext_pixels)
                int_std  = np.std(int_pixels) + 1e-6
                ext_std  = np.std(ext_pixels) + 1e-6

                # Scale exterior to match interior tone
                ext_matched = (img_ext_aligned[:, :, c].astype(np.float32) - ext_mean) \
                              * (int_std / ext_std) + int_mean
            else:
                ext_matched = img_ext_aligned[:, :, c].astype(np.float32)

            # Blend interior + tone-matched exterior
            bm = blend_mask[:, :, np.newaxis][:, :, 0] if blend_mask.ndim == 3 \
                 else blend_mask
            result_f[:, :, c] = (
                img_int[:, :, c].astype(np.float32) * (1 - bm) +
                ext_matched * bm
            )

        result = np.clip(result_f, 0, 255).astype(np.uint8)

        # Coverage stats
        window_pct = float(np.sum(blend_mask > 0.1)) / (h * w) * 100

        return {
            "success":      True,
            "image":        cv2_to_b64(result),
            "window_pct":   round(window_pct, 1),
            "aligned":      True,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"HDR blend failed: {str(e)}"
        )


# ── MULTI-EXPOSURE HDR BLEND ──────────────────────────────────
class MultiHDRRequest(BaseModel):
    images: list[str]          # list of base64 images, up to 7
    timestamps: list[str] = [] # optional EXIF timestamps per image
    blend_strength: float = 90.0
    feather: float = 40.0


@app.post("/hdr-blend-multi")
async def hdr_blend_multi(req: MultiHDRRequest):
    """
    Blend up to 7 bracketed exposures using Mertens exposure fusion.
    
    Mertens fusion picks the best-exposed pixels from each image:
    - Well-exposed (not blown out, not too dark) = high weight
    - Blown out or underexposed = low weight
    No tone mapping artifacts, natural looking result.
    """
    try:
        if len(req.images) < 2:
            raise ValueError("Need at least 2 images to blend")
        if len(req.images) > 7:
            req.images = req.images[:7]

        # Decode all images
        imgs = []
        MAX_DIM = 2000
        target_w = target_h = None

        for b64 in req.images:
            img = b64_to_cv2(b64)
            h, w = img.shape[:2]
            if max(h, w) > MAX_DIM:
                scale = MAX_DIM / max(h, w)
                img = cv2.resize(img, (int(w*scale), int(h*scale)),
                                 interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]
            if target_w is None:
                target_w, target_h = w, h
            else:
                # Resize all to match first image dimensions
                if w != target_w or h != target_h:
                    img = cv2.resize(img, (target_w, target_h),
                                     interpolation=cv2.INTER_AREA)
            imgs.append(img)

        h, w = target_h, target_w

        # ── Step 1: Align all images to the first ─────────────
        grey_ref = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
        aligned  = [imgs[0]]

        for i, img in enumerate(imgs[1:], 1):
            try:
                grey_img    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria    = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               50, 1e-4)
                _, warp_matrix = cv2.findTransformECC(
                    grey_ref, grey_img, warp_matrix,
                    cv2.MOTION_TRANSLATION, criteria
                )
                aligned_img = cv2.warpAffine(
                    img, warp_matrix, (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                )
                aligned.append(aligned_img)
            except Exception:
                aligned.append(img)  # use as-is if alignment fails

        # ── Step 2: Sort by brightness (darkest to brightest) ─
        def mean_brightness(img):
            return float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

        aligned.sort(key=mean_brightness)

        # ── Step 3: Mertens exposure fusion ───────────────────
        # OpenCV's implementation of Mertens (2007) — best algorithm
        # for real-world HDR without tone mapping artifacts
        merge_mertens = cv2.createMergeMertens(
            contrast_weight=1.0,    # prefer well-contrasted pixels
            saturation_weight=1.0,  # prefer saturated pixels
            exposure_weight=0.0     # don't weight by exposure level
        )
        fusion = merge_mertens.process(aligned)

        # Convert from 0-1 float to 0-255 uint8
        result = np.clip(fusion * 255, 0, 255).astype(np.uint8)

        # ── Step 4: Gentle post-processing ────────────────────
        # Slight contrast boost to compensate for Mertens' tendency
        # to produce slightly flat results
        result_f = result.astype(np.float32)
        # S-curve: lift shadows slightly, hold highlights
        lut = np.array([
            min(255, int(i * 1.05 - 0.05 * (i/255)**2 * 255))
            for i in range(256)
        ], dtype=np.uint8)
        result = cv2.LUT(result, lut)

        # Coverage stats
        n_imgs = len(aligned)

        return {
            "success":    True,
            "image":      cv2_to_b64(result),
            "images_used": n_imgs,
            "aligned":    True,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Multi HDR blend failed: {str(e)}"
        )


# ── EXIF TIMESTAMP READER ─────────────────────────────────────
class EXIFRequest(BaseModel):
    images: list[str]   # list of base64 images
    filenames: list[str] = []


@app.post("/read-exif")
async def read_exif(req: EXIFRequest):
    """
    Read EXIF timestamps from images to enable auto-grouping.
    Returns timestamp for each image so frontend can group by time window.
    """
    try:
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS
        import io

        results = []
        for i, b64 in enumerate(req.images):
            try:
                if "," in b64:
                    b64 = b64.split(",")[1]
                img_bytes = base64.b64decode(b64)
                pil_img   = PILImage.open(io.BytesIO(img_bytes))
                exif_data = pil_img._getexif()

                timestamp = None
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag in ('DateTimeOriginal', 'DateTime', 'DateTimeDigitized'):
                            timestamp = str(value)
                            break

                results.append({
                    "index":     i,
                    "filename":  req.filenames[i] if i < len(req.filenames) else f"image_{i}",
                    "timestamp": timestamp,
                    "has_exif":  timestamp is not None,
                })
            except Exception:
                results.append({
                    "index":     i,
                    "filename":  req.filenames[i] if i < len(req.filenames) else f"image_{i}",
                    "timestamp": None,
                    "has_exif":  False,
                })

        return {"success": True, "images": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EXIF read failed: {str(e)}")
