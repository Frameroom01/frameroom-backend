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

app = FastAPI(title="Frameroom Backend", version="1.0.0")

# Allow requests from Frameroom frontend (Vercel + local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: base64 → numpy image ─────────────────────────────
def b64_to_cv2(b64_string: str) -> np.ndarray:
    """Decode base64 image string to OpenCV numpy array (BGR)."""
    # Strip data URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── Helper: numpy image → base64 ─────────────────────────────
def cv2_to_b64(img: np.ndarray, quality: int = 92) -> str:
    """Encode OpenCV numpy array (BGR) to base64 JPEG string."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ── Request models ────────────────────────────────────────────
class LensCorrectionRequest(BaseModel):
    image: str          # base64 encoded image
    distortion: float   # barrel/pincushion: -100 to 100 (neg=barrel, pos=pincushion)
    vertical: float     # vertical perspective: -100 to 100
    horizontal: float   # horizontal perspective: -100 to 100

class AutoCorrectRequest(BaseModel):
    image: str          # base64 encoded image


# ── Root health check ─────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Frameroom backend running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}


# ── LENS CORRECTION ───────────────────────────────────────────
@app.post("/lens-correction")
async def lens_correction(req: LensCorrectionRequest):
    """
    Apply lens distortion correction and perspective correction.

    distortion: negative = fix barrel distortion (most wide-angle lenses)
                positive = fix pincushion distortion (rare)
    vertical:   fix converging verticals (walls leaning in/out)
    horizontal: fix horizontal perspective
    """
    try:
        img = b64_to_cv2(req.image)
        h, w = img.shape[:2]
        result = img.copy()

        # ── Step 1: Radial distortion correction ──────────────
        if abs(req.distortion) > 0.5:
            # Map slider -100→100 to k1 coefficient
            # Typical barrel distortion k1 is around -0.1 to -0.4
            # Positive distortion slider = barrel fix = negative k1
            k1 = -(req.distortion / 100.0) * 0.35
            k2 = k1 * 0.15  # secondary coefficient (smaller effect)

            # Camera matrix — assume principal point at center
            cx, cy = w / 2.0, h / 2.0
            focal = max(w, h) * 0.85  # estimated focal length

            camera_matrix = np.array([
                [focal, 0,     cx],
                [0,     focal, cy],
                [0,     0,     1 ]
            ], dtype=np.float64)

            dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)

            # Compute optimal new camera matrix to minimize black borders
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 0.85, (w, h)
            )

            result = cv2.undistort(result, camera_matrix, dist_coeffs, None, new_camera_matrix)

            # Crop to ROI to remove black borders
            x, y, rw, rh = roi
            if rw > 0 and rh > 0:
                result = result[y:y+rh, x:x+rw]
                # Resize back to original dimensions
                result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # ── Step 2: Perspective correction (vertical/horizontal) ──
        if abs(req.vertical) > 0.5 or abs(req.horizontal) > 0.5:
            h2, w2 = result.shape[:2]

            # Vertical perspective: adjusts the top-vs-bottom width ratio
            # Positive = top wider (fix walls leaning in)
            # Negative = bottom wider (rare)
            vert_factor  = req.vertical   / 100.0 * 0.25
            horiz_factor = req.horizontal / 100.0 * 0.25

            # Source points (full image corners)
            src = np.float32([
                [0,  0 ],
                [w2, 0 ],
                [w2, h2],
                [0,  h2]
            ])

            # Destination points — shift corners to correct perspective
            v_shift = vert_factor  * w2 * 0.5
            h_shift = horiz_factor * h2 * 0.5

            dst = np.float32([
                [0  + v_shift + h_shift,  0 ],
                [w2 - v_shift + h_shift,  0 ],
                [w2 + v_shift - h_shift,  h2],
                [0  - v_shift - h_shift,  h2]
            ])

            M = cv2.getPerspectiveTransform(src, dst)
            result = cv2.warpPerspective(
                result, M, (w2, h2),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE
            )

        return {"success": True, "image": cv2_to_b64(result)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lens correction failed: {str(e)}")


# ── AUTO LENS CORRECTION ──────────────────────────────────────
@app.post("/auto-lens-correct")
async def auto_lens_correct(req: AutoCorrectRequest):
    """
    Automatically detect and correct lens distortion using line detection.
    Finds straight lines in the photo, measures deviation, calculates
    correction values, and returns both the corrected image and the
    detected slider values so the UI can show what was applied.
    """
    try:
        img = b64_to_cv2(req.image)
        h, w = img.shape[:2]

        # Convert to greyscale for line detection
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Canny edge detection — find sharp edges
        edges = cv2.Canny(grey, 50, 150, apertureSize=3)

        # Hough line detection — find long straight lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=120,
            minLineLength=w * 0.25,  # lines must be at least 25% of image width
            maxLineGap=30
        )

        detected_vertical   = 0.0
        detected_distortion = 0.0

        if lines is not None:
            # Separate near-vertical and near-horizontal lines
            v_lines = []  # near vertical (walls, door frames)
            h_lines = []  # near horizontal (ceiling/floor junctions)

            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    angle = 90.0
                else:
                    angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))

                # Near vertical (within 20° of vertical)
                if angle > 70:
                    v_lines.append(line[0])
                # Near horizontal (within 20° of horizontal)
                elif angle < 20:
                    h_lines.append(line[0])

            # ── Detect vertical convergence ──
            # If walls lean inward, left-side vertical lines tilt right
            # and right-side vertical lines tilt left
            if len(v_lines) >= 2:
                left_tilts  = []
                right_tilts = []
                cx = w / 2

                for x1, y1, x2, y2 in v_lines:
                    mid_x = (x1 + x2) / 2
                    if x2 != x1:
                        tilt = (x2 - x1) / max(abs(y2 - y1), 1)
                    else:
                        tilt = 0
                    if mid_x < cx:
                        left_tilts.append(tilt)
                    else:
                        right_tilts.append(tilt)

                if left_tilts and right_tilts:
                    avg_left  = np.mean(left_tilts)
                    avg_right = np.mean(right_tilts)
                    # Convergence: left tilts right (+) and right tilts left (-)
                    convergence = avg_left - avg_right
                    # Map to vertical slider -100→100
                    detected_vertical = float(np.clip(convergence * 400, -60, 60))

            # ── Detect barrel distortion ──
            # Look for horizontal lines that bow outward (barrel)
            if len(h_lines) >= 3:
                deviations = []
                for x1, y1, x2, y2 in h_lines:
                    # Measure how curved the line appears
                    # (simplified: measure y deviation from straight line)
                    if abs(x2 - x1) > w * 0.3:
                        mid_y = (y1 + y2) / 2
                        expected_y = y1 + (y2 - y1) * 0.5
                        deviations.append(mid_y - expected_y)

                if deviations:
                    avg_dev = np.mean(deviations)
                    detected_distortion = float(np.clip(avg_dev * 2, -40, 40))

        # Apply the detected corrections
        correction_req = LensCorrectionRequest(
            image=req.image,
            distortion=detected_distortion,
            vertical=detected_vertical,
            horizontal=0.0
        )
        correction_result = await lens_correction(correction_req)

        return {
            "success":    True,
            "image":      correction_result["image"],
            "detected": {
                "distortion": round(detected_distortion, 1),
                "vertical":   round(detected_vertical, 1),
                "horizontal": 0.0,
            },
            "lines_found": len(lines) if lines is not None else 0,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto correction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
