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

            # Step 2: Inpaint black corners with content-aware fill
            # This is OpenCV's equivalent of Photoshop's content-aware fill
            grey_r = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, black_mask = cv2.threshold(grey_r, 10, 255, cv2.THRESH_BINARY_INV)

            # Only inpaint actual corner areas — not real dark content in the scene
            # Build a corner-only mask by excluding the center region
            corner_only = np.zeros((h2, w2), dtype=np.uint8)
            # Mark the four corner triangles based on warp shift amount
            corner_size = int(abs(v_shift) * 2.5 + abs(h_shift) * 2.5 + 20)
            corner_size = max(20, min(corner_size, int(min(h2, w2) * 0.35)))
            # Top-left and top-right corners
            cv2.fillPoly(corner_only, [np.array([[0,0],[corner_size,0],[0,corner_size]])], 255)
            cv2.fillPoly(corner_only, [np.array([[w2,0],[w2-corner_size,0],[w2,corner_size]])], 255)
            # Bottom-left and bottom-right corners
            cv2.fillPoly(corner_only, [np.array([[0,h2],[corner_size,h2],[0,h2-corner_size]])], 255)
            cv2.fillPoly(corner_only, [np.array([[w2,h2],[w2-corner_size,h2],[w2,h2-corner_size]])], 255)

            # Intersect: only fill areas that are both black AND in corners
            fill_mask = cv2.bitwise_and(black_mask, corner_only)

            # Dilate slightly to catch edge pixels
            kernel = np.ones((3, 3), np.uint8)
            fill_mask = cv2.dilate(fill_mask, kernel, iterations=2)

            if np.sum(fill_mask > 0) > 10:
                # INPAINT_TELEA — fast, high quality, content-aware
                result = cv2.inpaint(result, fill_mask, inpaintRadius=8,
                                     flags=cv2.INPAINT_TELEA)

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
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_vertical   = 0.0
        detected_distortion = 0.0
        lines_found = 0
        strategy_used = "none"

        # Strategy 1: Hough lines
        edges = cv2.Canny(grey, 25, 90, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=w*0.10, maxLineGap=60)

        if lines is not None:
            lines_found = len(lines)
            v_lines, h_lines = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = 90.0 if x2==x1 else abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                if angle > 68:   v_lines.append(line[0])
                elif angle < 22: h_lines.append(line[0])

            if len(v_lines) >= 3:
                left_t, right_t = [], []
                cx = w / 2
                for x1, y1, x2, y2 in v_lines:
                    tilt = (x2-x1) / max(abs(y2-y1), 1)
                    (left_t if (x1+x2)/2 < cx else right_t).append(tilt)
                if left_t and right_t:
                    conv = np.mean(left_t) - np.mean(right_t)
                    # Negate: positive convergence needs negative correction
                    detected_vertical = float(np.clip(-conv * 600, -65, 65))
                    strategy_used = "hough"

            if len(h_lines) >= 2:
                devs = [((y1+y2)/2)-(y1+(y2-y1)*0.5)
                        for x1,y1,x2,y2 in h_lines if abs(x2-x1) > w*0.15]
                if devs:
                    detected_distortion = float(np.clip(np.mean(devs)*3, -45, 45))

        # Strategy 2: Sobel fallback for exteriors
        if abs(detected_vertical) < 3:
            sx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(sx**2 + sy**2)
            vert_mask = (np.abs(sy) > np.abs(sx)*2) & (mag > np.percentile(mag, 75))

            if np.sum(vert_mask) > 100:
                ys, xs = np.where(vert_mask)
                lm = xs < w//2
                rm = xs >= w//2

                def lean(strip_xs, strip_ys):
                    tm = strip_ys < h//2
                    bm = strip_ys >= h//2
                    if np.sum(tm) < 5 or np.sum(bm) < 5: return 0
                    return float(np.mean(strip_xs[tm]) - np.mean(strip_xs[bm]))

                if np.sum(lm) > 20 and np.sum(rm) > 20:
                    ll = lean(xs[lm], ys[lm])
                    rl = lean(xs[rm], ys[rm])
                    # Negate: positive convergence needs negative correction
                    v_sobel = float(np.clip(-(ll - rl) / w * 120, -50, 50))
                    if abs(v_sobel) > abs(detected_vertical):
                        detected_vertical = v_sobel
                        strategy_used = "sobel"

        correction_req = LensCorrectionRequest(
            image=req.image,
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
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto correction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
