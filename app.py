"""
app.py — Apple Rootstock Image Analysis · Streamlit App
=========================================================
Run with:
    streamlit run app.py

Requires all .pth model files and score_cold_damage.py / predict_quality.py
to be in the same directory as this file.
"""

import io
import sys
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Apple Rootstock Analysis",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark background */
.stApp { background-color: #0f1117; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2a3040;
}

/* Score badge */
.score-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.8rem;
    font-weight: 600;
    padding: 0.15em 0.5em;
    border-radius: 6px;
    margin-bottom: 0.4em;
    letter-spacing: -0.02em;
}
.badge-green  { background: #0d3320; color: #2ecc71; border: 1px solid #2ecc71; }
.badge-yellow { background: #332100; color: #f39c12; border: 1px solid #f39c12; }
.badge-red    { background: #330d0d; color: #e74c3c; border: 1px solid #e74c3c; }

/* Prediction label pill */
.pred-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    padding: 0.2em 0.7em;
    border-radius: 4px;
    margin-bottom: 0.5em;
}
.pill-good   { background: #0d3320; color: #2ecc71; border: 1px solid #2ecc71; }
.pill-bad    { background: #330d0d; color: #e74c3c; border: 1px solid #e74c3c; }
.pill-review { background: #332100; color: #f39c12; border: 1px solid #f39c12; }

/* Stat cards */
.stat-card {
    background: #161b27;
    border: 1px solid #2a3040;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
}
.stat-label { font-size: 0.72rem; color: #6b7a99; letter-spacing: 0.08em; text-transform: uppercase; }
.stat-value { font-size: 1.4rem; font-weight: 600; color: #e0e6f0; margin-top: 0.1em; }

/* Section header rule */
.section-rule {
    border: none;
    border-top: 1px solid #2a3040;
    margin: 1.5rem 0 1rem 0;
}

/* Confidence bar */
.conf-bar-bg {
    background: #1e2535;
    border-radius: 4px;
    height: 8px;
    width: 100%;
    margin-top: 4px;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 4px;
    transition: width 0.3s;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers: model loading (cached) ──────────────────────────────────────────

@st.cache_resource
def load_classifier(weights_path: str):
    """Load an EfficientNet-B0 classifier from a .pth file."""
    import torch
    import torch.nn as nn
    import torchvision.models as models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = models.efficientnet_b0(weights=None)
    in_f   = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_f, 2),
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


@st.cache_resource
def get_transform():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Helpers: inference ───────────────────────────────────────────────────────

def run_classifier(pil_image, model, device, transform) -> tuple[float, float]:
    """Returns (prob_class0, prob_class1) for a PIL image."""
    import torch
    tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    return float(probs[0]), float(probs[1])


def segment_stem(bgr: np.ndarray) -> np.ndarray:
    """Teal-background segmentation — returns binary mask (255=stem)."""
    hsv    = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S   = hsv[:, :, 0], hsv[:, :, 1]
    bg     = (H >= 85) & (H <= 125) & (S > 15)
    fg     = (~bg).astype(np.uint8) * 255
    k      = max(5, bgr.shape[1] // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg     = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=4)
    fg     = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return np.ones(bgr.shape[:2], dtype=np.uint8) * 255
    areas   = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    mask    = (labels == largest).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull_mask = np.zeros_like(mask)
        hull      = cv2.convexHull(max(contours, key=cv2.contourArea))
        cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
        coverage = hull_mask.sum() / 255 / hull_mask.size
        if 0.05 < coverage < 0.95:
            return hull_mask
    return np.ones(bgr.shape[:2], dtype=np.uint8) * 255


def extract_features(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """18-dimensional feature vector for cold damage scoring."""
    from skimage.feature import graycomatrix, graycoprops

    mask_bool = mask.astype(bool)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    L, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    n_stem  = mask_bool.sum()

    brown      = (L < 200) & (a > 130) & (b > 128)
    brown_frac = (brown & mask_bool).sum() / n_stem if n_stem > 0 else 0.0

    L_v = L[mask_bool].astype(float); a_v = a[mask_bool].astype(float)
    b_v = b[mask_bool].astype(float)
    h_v = hsv[:,:,0][mask_bool].astype(float)
    s_v = hsv[:,:,1][mask_bool].astype(float)

    h_img, w_img = mask.shape
    cy, cx = h_img / 2, w_img / 2
    ys, xs = np.mgrid[0:h_img, 0:w_img]
    dist_c = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    r50    = np.percentile(dist_c[mask_bool], 50)
    inner  = mask_bool & (dist_c <= r50)
    outer  = mask_bool & (dist_c > r50)
    inner_frac = (brown & inner).sum() / inner.sum() if inner.sum() > 0 else 0.0
    outer_frac = (brown & outer).sum() / outer.sum() if outer.sum() > 0 else 0.0
    offset = 0.0
    if (brown & mask_bool).sum() > 0:
        by, bx = np.where(brown & mask_bool)
        offset = np.sqrt((by.mean()-cy)**2 + (bx.mean()-cx)**2) / max(r50, 1)

    L_q  = (L.copy() / 4).astype(np.uint8)
    L_q[~mask_bool] = 0
    glcm = graycomatrix(L_q, distances=[1,3], angles=[0, np.pi/2],
                        levels=64, symmetric=True, normed=True)
    c1 = float(graycoprops(glcm, "contrast").mean())
    h1 = float(graycoprops(glcm, "homogeneity").mean())

    return np.array([
        brown_frac,
        L_v.mean(), L_v.std(), a_v.mean(), a_v.std(), b_v.mean(), b_v.std(),
        h_v.mean(), h_v.std(), s_v.mean(), s_v.std(),
        inner_frac, outer_frac, offset,
        c1, h1, c1, h1,
    ])


@st.cache_resource
def load_anchor_features(anchor_dir: str):
    """Featurise all anchor images and return (scaled_feats, scores, scaler)."""
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import cdist

    ANCHOR_LABELS = {
        "004-sec_0100": 1, "003-sec_0147": 2, "011-sec_0155": 3,
        "032-sec_0176": 4, "029-sec_0653": 5, "004-sec_0292": 6,
        "030-FREC-0270": 7, "012-sec_0348": 8, "031-AFRS_0559": 9,
        "041-FREC_0281": 10,
    }
    anchor_path = Path(anchor_dir)
    file_map    = {p.stem.lower(): p for p in anchor_path.rglob("*")
                   if p.suffix.lower() in {".png",".jpg",".jpeg",".tif"}}

    feats, scores = [], []
    for stem, score in sorted(ANCHOR_LABELS.items(), key=lambda x: x[1]):
        p = file_map.get(stem.lower())
        if p is None:
            continue
        bgr  = cv2.imread(str(p))
        if bgr is None:
            continue
        scale = 1024 / bgr.shape[1]
        bgr   = cv2.resize(bgr, (1024, int(bgr.shape[0]*scale)), interpolation=cv2.INTER_AREA)
        mask  = segment_stem(bgr)
        feats.append(extract_features(bgr, mask))
        scores.append(score)

    if len(feats) < 2:
        return None, None, None

    feats  = np.array(feats)
    scaler = StandardScaler().fit(feats)
    return scaler.transform(feats), np.array(scores, dtype=float), scaler


def predict_damage_score(bgr: np.ndarray, anchor_feats, anchor_scores, scaler, k=3):
    from scipy.spatial.distance import cdist
    scale = 1024 / bgr.shape[1]
    bgr   = cv2.resize(bgr, (1024, int(bgr.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    mask  = segment_stem(bgr)
    feat  = extract_features(bgr, mask)
    feat_s = scaler.transform(feat.reshape(1, -1))
    dists  = cdist(feat_s, anchor_feats, metric="euclidean")[0]
    k_eff  = min(k, len(anchor_scores))
    idx    = np.argsort(dists)[:k_eff]
    nd, ns = dists[idx], anchor_scores[idx]
    eps    = 1e-6
    w      = 1.0 / (nd + eps); w /= w.sum()
    score_f = float(np.clip(np.dot(w, ns), 1, 10))
    conf    = float(1.0 / (1.0 + nd.mean()))
    return int(round(score_f)), score_f, conf, mask


def render_stem_overlay(pil_image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Return image with background dimmed to highlight stem region."""
    bgr    = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    scale  = 1024 / bgr.shape[1]
    small  = cv2.resize(bgr, (1024, int(bgr.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    dim    = (small * 0.25).astype(np.uint8)
    result = dim.copy()
    result[mask.astype(bool)] = small[mask.astype(bool)]
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


# ── UI helpers ────────────────────────────────────────────────────────────────

def stat_card(label: str, value: str):
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
    </div>""", unsafe_allow_html=True)


def conf_bar(conf: float, color: str):
    pct = int(conf * 100)
    st.markdown(f"""
    <div style="margin-top:6px">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:#6b7a99;">
            CONFIDENCE
        </span>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div>
        </div>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:#e0e6f0;">
            {conf:.1%}
        </span>
    </div>""", unsafe_allow_html=True)


def score_badge(score: int) -> str:
    cls = "badge-green" if score <= 3 else ("badge-yellow" if score <= 6 else "badge-red")
    return f'<span class="score-badge {cls}">{score}</span>'


def pred_pill(pred: str) -> str:
    if pred == "Y":
        return '<span class="pred-pill pill-good">✓ Good quality</span>'
    elif pred == "N":
        return '<span class="pred-pill pill-bad">✗ Poor quality</span>'
    else:
        return '<span class="pred-pill pill-review">⚠ Needs review</span>'


# ── Sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🍎 Rootstock Analysis")
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

    page = st.radio(
        "Tool",
        ["Quality Check", "Defect Detection", "Cold Damage Score", "About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown(
        "<span style='font-size:0.75rem;color:#6b7a99;font-family:IBM Plex Mono,monospace;'>"
        "MODELS ON DISK</span>",
        unsafe_allow_html=True,
    )
    for fname, label in [
        ("apple_quality_classifier.pth",  "Quality  (Y/N)"),
        ("apple_blurry_classifier.pth",   "Blurry defect"),
        ("apple_shadow_classifier.pth",   "Shadow defect"),
    ]:
        exists = Path(fname).exists()
        icon   = "🟢" if exists else "🔴"
        st.markdown(
            f"<span style='font-size:0.8rem;color:#aaa;font-family:IBM Plex Mono,monospace;'>"
            f"{icon} {label}</span>",
            unsafe_allow_html=True,
        )

    anchor_count = len(list(Path("anchors").rglob("*.png"))) if Path("anchors").exists() else 0
    st.markdown(
        f"<span style='font-size:0.8rem;color:#aaa;font-family:IBM Plex Mono,monospace;'>"
        f"{'🟢' if anchor_count >= 2 else '🔴'} {anchor_count}/10 anchors</span>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Quality Check
# ════════════════════════════════════════════════════════════════════════════
if page == "Quality Check":
    st.markdown("## Image Quality Check")
    st.markdown(
        "Upload one or more cross-section images. The model will classify each "
        "as **good quality (Y)**, **poor quality (N)**, or flag it for **manual review**."
    )

    model_path = "apple_quality_classifier.pth"
    if not Path(model_path).exists():
        st.error(f"`{model_path}` not found. Train the model first.")
        st.stop()

    c1, c2 = st.columns([2, 1])
    with c2:
        good_t  = st.slider("Good threshold", 0.5, 1.0, 0.75, 0.05,
                            help="prob_good ≥ this → Y")
        bad_t   = st.slider("Bad threshold",  0.0, 0.9, 0.40, 0.05,
                            help="prob_good < this → N; in between → REVIEW")
        if bad_t > good_t:
            st.warning("Bad threshold must be ≤ Good threshold.")
            st.stop()

    with c1:
        files = st.file_uploader(
            "Upload images", type=["png","jpg","jpeg","tif","tiff"],
            accept_multiple_files=True,
        )

    if not files:
        st.info("Upload at least one image to begin.")
        st.stop()

    model, device = load_classifier(model_path)
    transform     = get_transform()

    results = []
    for f in files:
        pil = Image.open(f).convert("RGB")
        p0, p1 = run_classifier(pil, model, device, transform)
        if p1 >= good_t:
            pred, col = "Y", "#2ecc71"
        elif p1 < bad_t:
            pred, col = "N", "#e74c3c"
        else:
            pred, col = "REVIEW", "#f39c12"
        results.append({"name": f.name, "pil": pil, "pred": pred,
                         "prob_good": p1, "prob_bad": p0, "col": col})

    # Summary bar
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1: stat_card("Total", str(len(results)))
    with sc2: stat_card("Good (Y)", str(sum(r["pred"]=="Y" for r in results)))
    with sc3: stat_card("Poor (N)", str(sum(r["pred"]=="N" for r in results)))
    with sc4: stat_card("Review",   str(sum(r["pred"]=="REVIEW" for r in results)))

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

    # Per-image results
    for r in results:
        with st.expander(f"{r['name']}  —  {r['pred']}", expanded=True):
            img_col, res_col = st.columns([1, 1])
            with img_col:
                st.image(r["pil"], use_container_width=True)
            with res_col:
                st.markdown(pred_pill(r["pred"]), unsafe_allow_html=True)
                conf_bar(max(r["prob_good"], r["prob_bad"]), r["col"])
                st.markdown("<br>", unsafe_allow_html=True)
                stat_card("Prob good", f"{r['prob_good']:.1%}")
                stat_card("Prob bad",  f"{r['prob_bad']:.1%}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Defect Detection
# ════════════════════════════════════════════════════════════════════════════
elif page == "Defect Detection":
    st.markdown("## Defect Detection")
    st.markdown(
        "Run the blurry or shadow defect classifier on uploaded images. "
        "Both models use the same EfficientNet-B0 architecture — only the "
        "training labels differ."
    )

    defect = st.radio("Defect type", ["Blurry", "Shadow"], horizontal=True)
    model_path = f"apple_{defect.lower()}_classifier.pth"

    if not Path(model_path).exists():
        st.error(f"`{model_path}` not found.")
        st.stop()

    threshold = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05,
                          help="prob_defect ≥ this → defect detected")

    files = st.file_uploader(
        "Upload images", type=["png","jpg","jpeg","tif","tiff"],
        accept_multiple_files=True, key="defect_upload",
    )

    if not files:
        st.info("Upload at least one image to begin.")
        st.stop()

    model, device = load_classifier(model_path)
    transform     = get_transform()

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

    flagged = 0
    for f in files:
        pil     = Image.open(f).convert("RGB")
        p0, p1  = run_classifier(pil, model, device, transform)  # p1 = defect prob
        detected = p1 >= threshold
        if detected:
            flagged += 1
        col_str = "#e74c3c" if detected else "#2ecc71"
        label   = f"⚠ {defect} detected" if detected else f"✓ No {defect.lower()} detected"

        with st.expander(f"{f.name}  —  {label}", expanded=True):
            ic, rc = st.columns([1, 1])
            with ic:
                st.image(pil, use_container_width=True)
            with rc:
                pill_cls = "pill-bad" if detected else "pill-good"
                st.markdown(
                    f'<span class="pred-pill {pill_cls}">{label}</span>',
                    unsafe_allow_html=True,
                )
                conf_bar(max(p0, p1), col_str)
                st.markdown("<br>", unsafe_allow_html=True)
                stat_card(f"Prob {defect.lower()}", f"{p1:.1%}")
                stat_card("Prob clean", f"{p0:.1%}")

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1: stat_card("Flagged", str(flagged))
    with sc2: stat_card("Clean",   str(len(files) - flagged))


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Cold Damage Score
# ════════════════════════════════════════════════════════════════════════════
elif page == "Cold Damage Score":
    st.markdown("## Cold Damage Scorer")
    st.markdown(
        "Assigns a freeze-injury score from **1** (no damage) to **10** (severe) "
        "based on tissue browning in stem cross-section images. "
        "Requires anchor images in the `anchors/` folder."
    )

    if not Path("anchors").exists() or anchor_count < 2:
        st.error("Need at least 2 anchor images in the `anchors/` folder.")
        st.stop()

    c1, c2 = st.columns([3, 1])
    with c2:
        k       = st.slider("Nearest anchors (k)", 1, min(anchor_count, 5), 3)
        show_seg = st.checkbox("Show segmentation overlay", value=True)
        skip_blurry = False
        if Path("apple_blurry_classifier.pth").exists():
            skip_blurry = st.checkbox("Skip blurry images", value=True)

    with c1:
        files = st.file_uploader(
            "Upload images", type=["png","jpg","jpeg","tif","tiff"],
            accept_multiple_files=True, key="score_upload",
        )

    if not files:
        st.info("Upload at least one image to begin.")
        st.stop()

    anchor_feats, anchor_scores, scaler = load_anchor_features("anchors")
    if anchor_feats is None:
        st.error("Could not load anchor features. Check the anchors/ folder.")
        st.stop()

    blurry_model, blurry_device = None, None
    if skip_blurry and Path("apple_blurry_classifier.pth").exists():
        blurry_model, blurry_device = load_classifier("apple_blurry_classifier.pth")
    transform = get_transform()

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

    scores_out = []
    for f in files:
        pil = Image.open(f).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # Blurry check
        if blurry_model is not None:
            _, p_blurry = run_classifier(pil, blurry_model, blurry_device, transform)
            if p_blurry >= 0.5:
                with st.expander(f"{f.name}  —  skipped (blurry)", expanded=False):
                    st.image(pil, width=300)
                    st.warning("Image flagged as blurry — not scored.")
                continue

        score_int, score_raw, conf, mask = predict_damage_score(
            bgr, anchor_feats, anchor_scores, scaler, k=k
        )
        scores_out.append(score_int)

        col_str = "#2ecc71" if score_int <= 3 else ("#f39c12" if score_int <= 6 else "#e74c3c")
        dmg_lbl = "Low damage" if score_int <= 3 else ("Moderate damage" if score_int <= 6 else "Severe damage")

        with st.expander(f"{f.name}  —  Score {score_int}  ({dmg_lbl})", expanded=True):
            ic, rc = st.columns([1, 1])
            with ic:
                if show_seg:
                    overlay = render_stem_overlay(pil, mask)
                    st.image(overlay, caption="Stem region highlighted",
                             use_container_width=True)
                else:
                    st.image(pil, use_container_width=True)

            with rc:
                st.markdown(score_badge(score_int), unsafe_allow_html=True)
                st.markdown(
                    f"<span style='color:#6b7a99;font-size:0.85rem;font-family:IBM Plex Mono,monospace;'>"
                    f"{dmg_lbl}</span>",
                    unsafe_allow_html=True,
                )
                conf_bar(conf, col_str)
                st.markdown("<br>", unsafe_allow_html=True)
                stat_card("Raw score",  f"{score_raw:.2f}")
                stat_card("Confidence", f"{conf:.3f}")

                # Mini progress bar for score
                st.markdown(
                    f"<div style='margin-top:12px'>"
                    f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;"
                    f"color:#6b7a99;letter-spacing:0.08em;'>DAMAGE SCALE  1 ─── 10</span>"
                    f"<div class='conf-bar-bg' style='margin-top:6px'>"
                    f"<div class='conf-bar-fill' style='width:{(score_int-1)/9*100:.0f}%;"
                    f"background:{col_str};'></div></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    if scores_out:
        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1: stat_card("Images scored", str(len(scores_out)))
        with sc2: stat_card("Mean score",    f"{np.mean(scores_out):.1f}")
        with sc3: stat_card("Range",         f"{min(scores_out)} – {max(scores_out)}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — About
# ════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("## About")
    st.markdown("""
This app provides three image analysis tools for apple rootstock stem cross-sections:

| Tool | Model | Task |
|------|-------|------|
| **Quality Check** | EfficientNet-B0 | Classify images as good (Y), poor (N), or uncertain (REVIEW) |
| **Defect Detection** | EfficientNet-B0 | Detect blurry or shadow artifacts specifically |
| **Cold Damage Score** | KNN + 18 CV features | Score freeze injury severity 1–10 based on tissue browning |

### Models
All three classifiers share the same **EfficientNet-B0** backbone pretrained on ImageNet,
with a `Dropout(0.3) → Linear(1280→2)` head fine-tuned on labeled rootstock images.
Weights are stored as PyTorch state dictionaries (`.pth` files).

The cold damage scorer is a **computer vision pipeline** — no neural network training required.
It segments the stem from the teal background using HSV thresholding, extracts an
18-dimensional feature vector (browning fraction, Lab/HSV colour statistics, spatial
distribution, GLCM texture), then scores via **inverse-distance weighted KNN**
against up to 10 manually labeled anchor images.

### Confidence
- **Classifier confidence** — `max(prob_good, prob_bad)` from the softmax output.
- **Damage score confidence** — `1 / (1 + mean_distance_to_k_anchors)`.
  Currently low (~0.18) because 8 anchors are sparse in 18-dimensional feature space.
  Will improve as more anchors are added.

### Retraining
Models can be retrained at any time using the training scripts in the repository.
New images in the existing folder structure are picked up automatically — no code changes needed.
""")

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#6b7a99;font-size:0.8rem;font-family:IBM Plex Mono,monospace;'>"
        "Apple Rootstock Image Analysis Pipeline</span>",
        unsafe_allow_html=True,
    )
