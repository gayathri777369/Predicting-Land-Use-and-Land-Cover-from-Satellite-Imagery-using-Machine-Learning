# app.py
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import joblib
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import torch.nn.functional as F
import pandas as pd

# -------------------------
# Classification function
# -------------------------
def classify_with_patches_latest(image_path, mobilenet, efficientnet, resnet50, meta_clf, class_names, device,
                        tile_size=256, threshold=20, knowledge_base_suggestions=None, transform=None,
                        return_fig=True):

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape

    predictions = []
    mobilenet.eval(); efficientnet.eval(); resnet50.eval()

    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            if i + tile_size <= h and j + tile_size <= w:
                patch = img.crop((j, i, j+tile_size, i+tile_size))
                patch_tensor = transform(patch).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs_mobilenet = F.softmax(mobilenet(patch_tensor), dim=1).cpu().numpy()
                    probs_efficientnet = F.softmax(efficientnet(patch_tensor), dim=1).cpu().numpy()
                    probs_resnet50 = F.softmax(resnet50(patch_tensor), dim=1).cpu().numpy()

                stacked_patch = np.concatenate([probs_mobilenet, probs_efficientnet, probs_resnet50], axis=1)
                pred_class_idx = meta_clf.predict(stacked_patch)[0]
                pred_class_name = class_names[pred_class_idx]
                predictions.append(pred_class_name)

    total_tiles = len(predictions)
    if total_tiles == 0:
        return {}, None, None

    class_counts = Counter(predictions)
    class_percentages = {cls: (count / total_tiles) * 100 for cls, count in class_counts.items()}

    max_percentage = max(class_percentages.values())
    top_classes = [cls for cls, pct in class_percentages.items() if pct == max_percentage]

    # Suggestions for all present classes
    suggestions_out = []
    if knowledge_base_suggestions is not None:
        present_classes = class_percentages.keys()
        for cls in present_classes:
            suggestions_out.extend(knowledge_base_suggestions.get(cls, ["No suggestion available."]))
        suggestions_out = list(dict.fromkeys(suggestions_out))  # deduplicate

    # Visualization (image + pie chart)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title(f"Dominant: {', '.join(top_classes)} ({max_percentage:.1f}%)")

    labels = list(class_percentages.keys())
    sizes = list(class_percentages.values())
    explode = [0.05 if cls in top_classes else 0 for cls in labels]

    axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
                explode=explode, textprops={'fontsize': 9}, shadow=True)
    axes[1].set_title("Class Coverage (%)")
    plt.tight_layout()

    if return_fig:
        return class_percentages, suggestions_out, fig
    return class_percentages, suggestions_out, None

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="LULC Classifier", layout="wide")
st.title("🌍 Land-Use Land-Cover Patch-wise Classifier")

# -------------------------
# Class Names
# -------------------------
CLASS_NAMES = [ "airplane", "airport", "baseball_diamond", "basketball_court", "beach", "bridge",
                "chaparral", "church", "circular_farmland", "cloud", "commercial_area",
                "dense_residential", "desert", "forest", "freeway", "golf_course",
                "ground_track_field", "harbor", "industrial_area", "intersection", "island",
                "lake", "meadow", "medium_residential", "mobile_home_park", "mountain",
                "overpass", "palace", "parking_lot", "railway", "railway_station",
                "rectangular_farmland", "river", "roundabout", "runway", "sea_ice",
                "ship", "snowberg", "sparse_residential", "stadium", "storage_tank",
                "tennis_court", "terrace", "thermal_power_station", "wetland" ]

MODEL_DIR = "models"

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_models(model_dir=MODEL_DIR, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(CLASS_NAMES)

    # 1. Recreate architectures
    mobilenet = models.mobilenet_v2(weights=None)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, num_classes)

    efficientnet = models.efficientnet_b0(weights=None)
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)

    resnet50 = models.resnet50(weights=None)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)

    # 2. Load weights
    mobilenet.load_state_dict(torch.load(os.path.join(model_dir, "mobilenet.pth"), map_location=device))
    efficientnet_ckpt = torch.load(os.path.join(model_dir, "efficientnet.pth"), map_location=device)
    efficientnet.load_state_dict(efficientnet_ckpt["model_state_dict"])
    resnet50.load_state_dict(torch.load(os.path.join(model_dir, "resnet50.pth"), map_location=device))

    # 3. Device + eval
    mobilenet.to(device).eval()
    efficientnet.to(device).eval()
    resnet50.to(device).eval()

    # 4. Meta classifier
    meta_clf = joblib.load(os.path.join(model_dir, "meta_clf.pkl"))

    return mobilenet, efficientnet, resnet50, meta_clf, device

try:
    mobilenet, efficientnet, resnet50, meta_clf, device = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# -------------------------
# Transform
# -------------------------
tile_size = 256
transform = T.Compose([
    T.Resize((tile_size, tile_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -------------------------
# Knowledge Base (example)
# -------------------------
KNOWLEDGE_BASE = {
   "airplane": ["Presence of airplanes detected – area may be near an airport or airstrip; consider aviation safety and noise management."],
    "airport": ["Airport detected – suitable for urban planning, transport infrastructure development, or logistics planning."],
    "baseball_diamond": ["Sports facility detected – could be utilized for recreation, community events, or sports management."],
    "basketball_court": ["Sports facility detected – suitable for local recreation, urban sports planning, or community development."],
    "beach": ["Beach area detected – opportunities for tourism, conservation, and coastal management."],
    "bridge": ["Bridge detected – consider transportation planning, structural maintenance, or traffic management."],
    "chaparral": ["Chaparral vegetation detected – land may be suitable for ecological conservation, wildfire management, or habitat protection."],
    "church": ["Religious structure detected – area could be considered for cultural heritage management or urban planning around community centers."],
    "circular_farmland": ["Circular agricultural fields detected – suitable for crop monitoring, irrigation planning, and agricultural optimization."],
    "cloud": ["Cloud coverage detected – may affect aerial or satellite imagery analysis; consider revisiting imagery on clear days."],
    "commercial_area": ["Commercial area detected – suitable for business development, urban planning, or economic analysis."],
    "dense_residential": ["High-density residential area – consider urban infrastructure, housing planning, and public services management."],
    "desert": ["Desert land detected – suitable for solar energy projects, desertification monitoring, or ecological studies."],
    "forest": ["Forested area detected – suitable for forestry, conservation, eco-tourism, or wildlife habitat management."],
    "freeway": ["Freeway detected – useful for transportation planning, traffic analysis, and infrastructure maintenance."],
    "golf_course": ["Golf course detected – potential for recreation planning, tourism, or landscape maintenance."],
    "ground_track_field": ["Athletic field detected – area may be used for sports events, community activities, or educational purposes."],
    "harbor": ["Harbor or port detected – suitable for shipping, logistics, trade, and coastal management."],
    "industrial_area": ["Industrial zone detected – relevant for industrial development, pollution monitoring, or urban planning."],
    "intersection": ["Road intersection detected – useful for traffic analysis, safety planning, and urban infrastructure improvement."],
    "island": ["Island detected – opportunities for tourism, conservation, and land-use planning."],
    "lake": ["Lake detected – suitable for water resource management, aquaculture, or recreational planning."],
    "meadow": ["Meadow detected – potential for grazing, ecological studies, or landscape conservation."],
    "medium_residential": ["Medium-density residential area – consider housing management, local amenities, and urban planning."],
    "mobile_home_park": ["Mobile home park detected – suitable for community planning, infrastructure, and social services assessment."],
    "mountain": ["Mountainous area detected – relevant for ecological conservation, tourism, and risk management (landslides, erosion)."],
    "overpass": ["Overpass detected – useful for transportation planning, safety assessments, and infrastructure maintenance."],
    "palace": ["Palace or historical building detected – area may be suitable for cultural heritage protection or tourism management."],
    "parking_lot": ["Parking facility detected – consider urban traffic management, commercial planning, or land-use optimization."],
    "railway": ["Railway detected – relevant for transportation planning, logistics, and connectivity analysis."],
    "railway_station": ["Railway station detected – opportunities for urban development, transport integration, and commuter services."],
    "rectangular_farmland": ["Rectangular agricultural fields detected – suitable for crop monitoring, irrigation management, and farm planning."],
    "river": ["River detected – potential for water resource management, flood monitoring, or ecological conservation."],
    "roundabout": ["Roundabout detected – useful for traffic analysis, urban planning, and safety management."],
    "runway": ["Runway detected – area may be part of an airport; consider aviation management, safety, and infrastructure planning."],
    "sea_ice": ["Sea ice detected – relevant for climate monitoring, maritime navigation, and environmental studies."],
    "ship": ["Ships detected – indicates water transport or port activity; consider maritime logistics or safety monitoring."],
    "snowberg": ["Snowberg/glacier detected – area may require environmental monitoring, climate research, or tourism planning."],
    "sparse_residential": ["Low-density residential area – consider housing development, urban expansion, or community planning."],
    "stadium": ["Stadium detected – suitable for sports management, events planning, or urban recreation planning."],
    "storage_tank": ["Storage facility detected – may indicate industrial activity; consider safety, environmental monitoring, or infrastructure management."],
    "tennis_court": ["Sports facility detected – area suitable for community recreation, urban planning, or sports events."],
    "terrace": ["Terraced land detected – suitable for agriculture, landscape management, or soil erosion control."],
    "thermal_power_station": ["Thermal power station detected – relevant for energy infrastructure, environmental monitoring, or urban planning."],
    "wetland": ["Wetland detected – potential for conservation, biodiversity protection, and water resource management."]
}

# -------------------------
# UI
# -------------------------
uploaded_file = st.file_uploader("Upload satellite image (jpg/png)", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    #st.image(image, caption="Uploaded image", use_container_width=True)
    tmp_path = "temp_upload.jpg"
    image.save(tmp_path)

    if st.button("Run patch-wise classification"):
        with st.spinner("Classifying patches..."):
            class_percents, suggestions, fig = classify_with_patches_latest(
                tmp_path, mobilenet, efficientnet, resnet50, meta_clf, CLASS_NAMES, device,
                tile_size=tile_size, threshold=20, transform=transform,
                knowledge_base_suggestions=KNOWLEDGE_BASE, return_fig=True
            )

        if not class_percents:
            st.warning("No valid tiles found (image too small?).")
        else:
            # 1️⃣ Class coverage
            st.subheader("📊 Class coverage")
            for cls, pct in sorted(class_percents.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{cls}**: {pct:.2f}%")

            # 2️⃣ Visualization
            st.subheader("Visualization")
            st.pyplot(fig)

            # 3️⃣ Suggestions
            if suggestions:
                st.subheader("💡 Suggestions / Insights")
                for s in suggestions:
                    st.write(f"- {s}")

            # 4️⃣ Download CSV
            df = pd.DataFrame(list(class_percents.items()), columns=["class","percentage"])
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV report", data=csv, file_name="class_coverage.csv", mime="text/csv")
