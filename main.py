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

        mask1 = soft_mask(tier1_clean, 31)
        mask2 = soft_mask(tier2_clean, 21)
        mask3 = soft_mask(tier3_clean, 15)
        mask_ceil = soft_mask(ceiling_clean, 35)  # wide feather for ceiling glow

        # Count regions
        n1, _, _, _ = cv2.connectedComponentsWithStats(tier1_clean, 8)
        n2, _, _, _ = cv2.connectedComponentsWithStats(tier2_clean, 8)
        nc, _, _, _ = cv2.connectedComponentsWithStats(ceiling_clean, 8)
        total_regions = max(0, (n1-1) + (n2-1) + (nc-1))
        coverage = float(np.sum(tier2_clean > 0)) / (h * w) * 100

        # ── PASS 2: Build reference using luminance-only approach ──
        # Key insight: preserve original hue/saturation, only reduce brightness
        # This prevents color bleeding from inpainting adjacent different-colored areas
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_ch, s_ch, v_ch = cv2.split(img_hsv)

        # Build a target value (brightness) channel
        # For glare areas: reduce V toward what the surface "should" be
        # Estimate non-glare brightness by blurring the V channel heavily
        # This gives us the ambient light level without the glare spike
        v_ambient = cv2.GaussianBlur(v_ch, (61, 61), 0)

        # For very bright pixels, the ambient is a better estimate of true brightness
        # Cap the reduction so we don't make areas too dark
        v_target = np.clip(v_ambient * 1.1, 0, 220)

        # Combined glare mask for HSV correction
        combined_mask = np.clip(mask1 * 1.0 + mask2 * 0.8 + mask3 * 0.5 + mask_ceil * 0.7, 0, 1)

        # Apply correction in HSV space — V channel only, keep H and S intact
        v_corrected = v_ch * (1 - combined_mask * strength) + v_target * (combined_mask * strength)
        v_corrected = np.clip(v_corrected, 0, 255)

        # Slight saturation reduction in glare areas (glare desaturates naturally)
        s_corrected = s_ch * (1 - combined_mask * strength * 0.3)
        s_corrected = np.clip(s_corrected, 0, 255)

        # Rebuild image in HSV then convert back to BGR
        corrected_hsv = cv2.merge([h_ch, s_corrected, v_corrected]).astype(np.uint8)
        result = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)

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
