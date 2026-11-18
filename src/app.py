
# SoilSenseAI — Punjab Soil Intelligence System
# Geospatial soil analysis and crop compatibility dashboard.

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import folium
from streamlit_folium import st_folium

from rasterio.warp import transform


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

APNA_PATH = (
    DATA_DIR
    / "ApnaFarmData"
    / "punjab_apna_farm_soil_dataset.csv"
)

APNA_SEARCH_LIMIT_KM = 5.0
RASTER_SEARCH_LIMIT_METRES = 3000

RASTERS = {
    "Soil pH": {
        "file": "punjab_phh2o_0-5cm_mean_clipped.tif",
        "scale": 10,
        "unit": "",
        "category": "Chemical",
    },
    "Organic Carbon": {
        "file": "punjab_soc_0-5cm_mean_clipped.tif",
        "scale": 10,
        "unit": "g/kg",
        "category": "Chemical",
    },
    "Nitrogen": {
        "file": "punjab_nitrogen_0-5cm_mean_clipped.tif",
        "scale": 100,
        "unit": "g/kg",
        "category": "Chemical",
    },
    "Clay": {
        "file": "punjab_clay_0-5cm_mean_clipped.tif",
        "scale": 10,
        "unit": "%",
        "category": "Physical",
    },
    "Sand": {
        "file": "punjab_sand_0-5cm_mean_clipped.tif",
        "scale": 10,
        "unit": "%",
        "category": "Physical",
    },
    "Silt": {
        "file": "punjab_silt_0-5cm_mean_clipped.tif",
        "scale": 10,
        "unit": "%",
        "category": "Physical",
    },
    "CEC": {
        "file": "punjab_cec_0-5cm_mean_clipped.tif",
        "scale": 10,
        "unit": "cmol(+)/kg",
        "category": "Chemical",
    },
    "Bulk Density": {
        "file": "punjab_bdod_0-5cm_mean_clipped.tif",
        "scale": 100,
        "unit": "kg/dm³",
        "category": "Physical",
    },
}


# =========================================================
# 2. PAGE CONFIGURATION AND DESIGN
# =========================================================

st.set_page_config(
    page_title="SoilSenseAI | Punjab Soil Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global canvas and typography */
    .stApp {
        background-color: #F3F6FA;
        color: #172B4D;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #E7EDF4;
        border-right: 1px solid #CBD5E1;
    }

    [data-testid="stSidebar"] * {
        color: #172B4D;
    }

    /* Main content text */
    .stApp p,
    .stApp li,
    .stApp label,
    .stMarkdown,
    [data-testid="stCaptionContainer"] {
        color: #24364B;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: #102A43;
    }

    /* Hero */
    .hero {
        background: linear-gradient(115deg, #102A43, #075985);
        padding: 30px 34px;
        border-radius: 15px;
        color: #FFFFFF;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 36px;
        margin-bottom: 5px;
    }

    .hero p {
        color: #EAF4FF !important;
        font-size: 15px;
        margin-bottom: 0;
    }

    .section-heading {
        color: #123B5D;
        font-size: 21px;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 15px;
    }

    .small-muted {
        color: #475569;
        font-size: 13px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        padding: 15px;
        border-radius: 10px;
    }

    [data-testid="stMetricLabel"] {
        color: #334155 !important;
    }

    [data-testid="stMetricValue"] {
        color: #102A43 !important;
    }

    /* Inputs */
    .stTextInput input,
    .stNumberInput input,
    [data-baseweb="input"] input {
        background-color: #FFFFFF !important;
        color: #102A43 !important;
        border: 1px solid #94A3B8 !important;
    }

    [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #102A43 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #334155 !important;
    }

    .stTabs [aria-selected="true"] {
        color: #075985 !important;
        border-bottom-color: #0284C7 !important;
    }

    /* Buttons */
    .stButton button {
        background-color: #075985 !important;
        color: #FFFFFF !important;
        border: 1px solid #075985 !important;
        border-radius: 8px;
    }

    .stButton button:hover {
        background-color: #0C4A6E !important;
        border-color: #0C4A6E !important;
        color: #FFFFFF !important;
    }

    /* Tables and containers */
    div[data-testid="stDataFrame"] {
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        overflow: hidden;
    }

    .status-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: #172B4D;
    }

    [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
    }

    .footer {
        text-align: center;
        color: #475569;
        font-size: 12px;
        padding-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. DATA LOADING
# =========================================================

@st.cache_data
def load_apna_data():

    if not APNA_PATH.exists():
        raise FileNotFoundError(
            f"APNA dataset not found: {APNA_PATH}"
        )

    df = pd.read_csv(APNA_PATH)

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    required = {"latitude", "longitude"}

    if not required.issubset(df.columns):
        raise ValueError(
            "Dataset must contain latitude and longitude."
        )

    df["latitude"] = pd.to_numeric(
        df["latitude"], errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"], errors="coerce"
    )

    return df.dropna(
        subset=["latitude", "longitude"]
    )


# =========================================================
# 4. GEOGRAPHIC DISTANCE
# =========================================================

def haversine_km(lat1, lon1, lat2, lon2):

    earth_radius = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2
        * earth_radius
        * np.arcsin(np.sqrt(a))
    )


# =========================================================
# 5. APNA SPATIAL LOOKUP
# =========================================================

def get_apna_location(df, latitude, longitude):

    distances = haversine_km(
        latitude,
        longitude,
        df["latitude"].to_numpy(),
        df["longitude"].to_numpy(),
    )

    nearest_index = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_index])

    if nearest_distance > APNA_SEARCH_LIMIT_KM:
        return None, nearest_distance

    return df.iloc[nearest_index], nearest_distance


def get_apna_value(row, property_name, field_type):

    property_name = property_name.lower()

    column_options = {
        "midpoint": [
            f"{property_name}_value_midpoint",
            f"{property_name}_midpoint",
        ],
        "minimum": [
            f"{property_name}_value_min",
            f"{property_name}_min",
        ],
        "maximum": [
            f"{property_name}_value_max",
            f"{property_name}_max",
        ],
        "class": [
            f"{property_name}_class_label",
            f"{property_name}_class",
            f"{property_name}_class_id",
        ],
    }

    for column in column_options.get(field_type, []):

        if column in row.index:

            value = row[column]

            if pd.notna(value):
                return value

    return None


def format_apna_property(row, property_name):

    minimum = get_apna_value(
        row, property_name, "minimum"
    )

    maximum = get_apna_value(
        row, property_name, "maximum"
    )

    midpoint = get_apna_value(
        row, property_name, "midpoint"
    )

    category = get_apna_value(
        row, property_name, "class"
    )

    parts = []

    if minimum is not None and maximum is not None:
        parts.append(f"{minimum} – {maximum}")

    elif midpoint is not None:
        parts.append(f"{midpoint}")

    if category is not None:
        parts.append(f"Class: {category}")

    return " | ".join(parts) if parts else "Not available"


# =========================================================
# 6. SOILGRIDS RASTER SAMPLING
# =========================================================

def sample_raster(
    raster_path,
    latitude,
    longitude,
    scale_factor,
    max_distance=RASTER_SEARCH_LIMIT_METRES,
):

    if not raster_path.exists():
        return {
            "value": None,
            "distance": None,
            "status": "File unavailable",
        }

    try:

        with rasterio.open(raster_path) as src:

            x, y = transform(
                "EPSG:4326",
                src.crs,
                [longitude],
                [latitude],
            )

            x = x[0]
            y = y[0]

            row, col = src.index(x, y)

            if not (
                0 <= row < src.height
                and 0 <= col < src.width
            ):
                return {
                    "value": None,
                    "distance": None,
                    "status": "Outside coverage",
                }

            target = src.read(
                1,
                window=rasterio.windows.Window(
                    col, row, 1, 1
                ),
                masked=True,
            )

            target_value = target[0, 0]

            # Use the exact coordinate pixel if valid.
            if (
                not np.ma.is_masked(target_value)
                and np.isfinite(float(target_value))
                and float(target_value) > 0
            ):

                return {
                    "value": float(target_value) / scale_factor,
                    "distance": 0.0,
                    "status": "Exact pixel",
                }

            # Search neighbouring pixels when target is NoData.
            pixel_size = min(
                abs(src.transform.a),
                abs(src.transform.e),
            )

            radius = int(
                np.ceil(max_distance / pixel_size)
            )

            col_start = max(0, col - radius)
            row_start = max(0, row - radius)

            col_end = min(src.width, col + radius + 1)
            row_end = min(src.height, row + radius + 1)

            window = rasterio.windows.Window(
                col_start,
                row_start,
                col_end - col_start,
                row_end - row_start,
            )

            data = src.read(
                1,
                window=window,
                masked=True,
            )

            # IMPORTANT: convert int16 masked data to float
            # before filling masked values with NaN.
            data = data.astype("float32")
            values = data.filled(np.nan)

            valid = (
                (~np.ma.getmaskarray(data))
                & np.isfinite(values)
                & (values > 0)
            )

            if not valid.any():

                return {
                    "value": None,
                    "distance": None,
                    "status": "No nearby valid pixel",
                }

            local_rows, local_cols = np.where(valid)

            global_rows = local_rows + row_start
            global_cols = local_cols + col_start

            pixel_x, pixel_y = src.xy(
                global_rows,
                global_cols,
                offset="center",
            )

            pixel_x = np.asarray(pixel_x)
            pixel_y = np.asarray(pixel_y)

            distances = np.sqrt(
                (pixel_x - x) ** 2
                + (pixel_y - y) ** 2
            )

            nearest_index = int(np.argmin(distances))

            nearest_distance = float(
                distances[nearest_index]
            )

            if nearest_distance > max_distance:

                return {
                    "value": None,
                    "distance": nearest_distance,
                    "status": "Beyond search radius",
                }

            raw_value = values[
                local_rows[nearest_index],
                local_cols[nearest_index],
            ]

            return {
                "value": float(raw_value) / scale_factor,
                "distance": nearest_distance,
                "status": "Nearest valid pixel",
            }

    except Exception as error:

        return {
            "value": None,
            "distance": None,
            "status": str(error),
        }


def get_soil_profile(latitude, longitude):

    profile = {}

    for property_name, config in RASTERS.items():

        result = sample_raster(
            raster_path=DATA_DIR / config["file"],
            latitude=latitude,
            longitude=longitude,
            scale_factor=config["scale"],
        )

        profile[property_name] = {
            **result,
            "unit": config["unit"],
            "category": config["category"],
        }

    return profile


# =========================================================
# 7. SOIL CONDITION ASSESSMENT
# =========================================================

def assess_soil(profile):

    findings = []

    ph = profile["Soil pH"]["value"]
    soc = profile["Organic Carbon"]["value"]

    if ph is not None:

        if ph > 8.5:

            findings.append({
                "Indicator": "Soil pH",
                "Finding": "Potential alkalinity",
                "Observation": f"Estimated pH: {ph:.2f}",
                "Priority": "Review",
            })

        elif ph < 5.5:

            findings.append({
                "Indicator": "Soil pH",
                "Finding": "Potential acidity",
                "Observation": f"Estimated pH: {ph:.2f}",
                "Priority": "Review",
            })

        else:

            findings.append({
                "Indicator": "Soil pH",
                "Finding": "Within broad screening range",
                "Observation": f"Estimated pH: {ph:.2f}",
                "Priority": "Normal",
            })

    if soc is not None:

        if soc < 5:

            findings.append({
                "Indicator": "Organic Carbon",
                "Finding": "Low estimated organic carbon",
                "Observation": f"{soc:.2f} g/kg",
                "Priority": "Review",
            })

        else:

            findings.append({
                "Indicator": "Organic Carbon",
                "Finding": "Above low-carbon threshold",
                "Observation": f"{soc:.2f} g/kg",
                "Priority": "Normal",
            })

    return findings


# =========================================================
# 8. CROP COMPATIBILITY
# =========================================================

CROP_PH_RANGES = {
    "Wheat": (6.0, 8.0),
    "Rice": (5.5, 7.5),
    "Cotton": (5.5, 8.0),
    "Maize": (5.8, 7.5),
    "Sugarcane": (6.0, 8.0),
}


def get_crop_compatibility(ph):

    results = []

    if ph is None:
        return results

    for crop, (minimum, maximum) in CROP_PH_RANGES.items():

        if minimum <= ph <= maximum:
            status = "Compatible"

        elif minimum - 0.5 <= ph <= maximum + 0.5:
            status = "Near boundary"

        else:
            status = "Outside range"

        results.append({
            "Crop": crop,
            "Preferred pH range": f"{minimum} – {maximum}",
            "Compatibility": status,
        })

    return results


# =========================================================
# 9. SIDEBAR — LOCATION INPUT
# =========================================================

st.sidebar.markdown("## 📍 Location Selection")
st.sidebar.caption(
    "Click the map to select a point in Punjab, or enter coordinates manually."
)

# Keep the selected coordinates between Streamlit reruns.
if "latitude" not in st.session_state:
    st.session_state.latitude = 30.67

if "longitude" not in st.session_state:
    st.session_state.longitude = 73.10

# Interactive map: a click updates the coordinate fields below.
selection_map = folium.Map(
    location=[
        st.session_state.latitude,
        st.session_state.longitude
    ],
    zoom_start=7,
    tiles="CartoDB positron",
    control_scale=True
)

folium.Marker(
    location=[
        st.session_state.latitude,
        st.session_state.longitude
    ],
    tooltip="Selected analysis location",
    popup=(
        f"Latitude: {st.session_state.latitude:.6f}<br>"
        f"Longitude: {st.session_state.longitude:.6f}"
    ),
    icon=folium.Icon(color="green", icon="map-marker")
).add_to(selection_map)

map_result = st_folium(
    selection_map,
    height=320,
    width=None,
    key="soil_location_picker",
    returned_objects=["last_clicked"]
)

if map_result and map_result.get("last_clicked"):
    clicked = map_result["last_clicked"]
    clicked_lat = float(clicked["lat"])
    clicked_lon = float(clicked["lng"])

    if (
        abs(clicked_lat - st.session_state.latitude) > 1e-7
        or abs(clicked_lon - st.session_state.longitude) > 1e-7
    ):
        st.session_state.latitude = clicked_lat
        st.session_state.longitude = clicked_lon
        st.rerun()

latitude = st.sidebar.number_input(
    "Latitude",
    min_value=27.0,
    max_value=35.0,
    step=0.001,
    format="%.6f",
    key="latitude",
)

longitude = st.sidebar.number_input(
    "Longitude",
    min_value=69.0,
    max_value=76.0,
    step=0.001,
    format="%.6f",
    key="longitude",
)

st.sidebar.caption(
    f"Selected point: {latitude:.5f}, {longitude:.5f}"
)

st.sidebar.divider()

analyze = st.sidebar.button(
    "Analyze Location",
    type="primary",
    width="stretch",
)

st.sidebar.markdown(
    """
    <div class="small-muted">
    <b>SoilSenseAI</b><br>
    Punjab Soil Intelligence Platform<br>
    Coordinate-based spatial analysis
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 10. HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌱 SoilSenseAI</h1>
        <p>Punjab Soil Intelligence System</p>
        <p>
        Geospatial soil profiling, soil condition assessment
        and crop compatibility analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 11. ANALYSIS STATE
# =========================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if analyze:
    if not (27.0 <= latitude <= 35.0 and 69.0 <= longitude <= 76.0):
        st.sidebar.error("Please select coordinates within the Punjab coverage area.")
    else:
        st.session_state.analysis_result = {
            "latitude": latitude,
            "longitude": longitude,
        }


# =========================================================
# 12. MAIN DASHBOARD
# =========================================================

if st.session_state.analysis_result is None:

    st.markdown("### Welcome to SoilSenseAI")

    st.write(
        """
        Analyze soil characteristics across Punjab using
        geographic coordinates.

        Click a location on the Punjab map in the sidebar,
        or enter latitude and longitude manually, then select
        Analyze Location.
        """
    )

else:

    selected_lat = st.session_state.analysis_result["latitude"]
    selected_lon = st.session_state.analysis_result["longitude"]

    try:
        apna_df = load_apna_data()

    except Exception as error:
        st.error(f"Dataset loading error: {error}")
        st.stop()

    apna_row, apna_distance = get_apna_location(
        apna_df,
        selected_lat,
        selected_lon,
    )

    profile = get_soil_profile(
        selected_lat,
        selected_lon,
    )

    available_properties = sum(
        item["value"] is not None
        for item in profile.values()
    )

    soil_findings = assess_soil(profile)

    ph_value = profile["Soil pH"]["value"]

    crop_results = get_crop_compatibility(ph_value)

    # -----------------------------------------------------
    # LOCATION SUMMARY
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-heading">Location Overview</div>',
        unsafe_allow_html=True,
    )

    loc1, loc2, loc3, loc4 = st.columns(4)

    loc1.metric(
        "Latitude",
        f"{selected_lat:.5f}",
    )

    loc2.metric(
        "Longitude",
        f"{selected_lon:.5f}",
    )

    loc3.metric(
        "Soil Properties",
        f"{available_properties}/8",
    )

    loc4.metric(
        "APNA Cell Distance",
        f"{apna_distance:.2f} km"
        if apna_row is not None
        else "Unavailable",
    )

    st.divider()

    # -----------------------------------------------------
    # DASHBOARD TABS
    # -----------------------------------------------------

    tab_overview, tab_profile, tab_crops, tab_method = st.tabs(
        [
            "📊 Overview",
            "🧪 Soil Profile",
            "🌾 Crop Compatibility",
            "🗺️ Methodology",
        ]
    )

    # =====================================================
    # TAB 1 — OVERVIEW
    # =====================================================

    with tab_overview:

        st.markdown(
            '<div class="section-heading">Soil Intelligence Summary</div>',
            unsafe_allow_html=True,
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Estimated Soil pH",
            f"{ph_value:.2f}" if ph_value is not None else "N/A",
        )

        soc_value = profile["Organic Carbon"]["value"]

        metric2.metric(
            "Organic Carbon",
            f"{soc_value:.2f} g/kg"
            if soc_value is not None
            else "N/A",
        )

        texture = "Not available"

        if apna_row is not None:

            texture_value = apna_row.get(
                "texture_class_label",
                None,
            )

            if pd.notna(texture_value):
                texture = str(texture_value)

        metric3.metric(
            "Soil Texture",
            texture,
        )

        st.markdown(
            '<div class="section-heading">Soil Condition Indicators</div>',
            unsafe_allow_html=True,
        )

        if soil_findings:

            findings_df = pd.DataFrame(soil_findings)

            st.dataframe(
                findings_df,
                hide_index=True,
                width="stretch",
            )

        else:

            st.info(
                "Soil condition indicators are unavailable "
                "for this location."
            )

        st.markdown(
            '<div class="section-heading">Geographic Location</div>',
            unsafe_allow_html=True,
        )

        map_df = pd.DataFrame({
            "latitude": [selected_lat],
            "longitude": [selected_lon],
        })

        st.map(map_df)

        st.caption(
            f"Analyzed coordinate: {selected_lat:.5f}, "
            f"{selected_lon:.5f}"
        )

    # =====================================================
    # TAB 2 — SOIL PROFILE
    # =====================================================

    with tab_profile:

        st.markdown(
            '<div class="section-heading">Soil Property Profile</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Topsoil depth: 0–5 cm"
        )

        profile_rows = []

        for property_name, result in profile.items():

            value = result["value"]

            if value is None:
                display_value = "Unavailable"

            else:
                display_value = (
                    f"{value:.2f} {result['unit']}".strip()
                )

            if result["distance"] is None:
                distance = "—"

            elif result["distance"] == 0:
                distance = "Exact pixel"

            else:
                distance = f"{result['distance']:.0f} m"

            profile_rows.append({
                "Property": property_name,
                "Category": result["category"],
                "Estimated Value": display_value,
                "Pixel Distance": distance,
            })

        profile_df = pd.DataFrame(profile_rows)

        st.dataframe(
            profile_df,
            hide_index=True,
            width="stretch",
        )

        st.markdown(
            '<div class="section-heading">APNA Farm Soil Information</div>',
            unsafe_allow_html=True,
        )

        if apna_row is not None:

            st.metric(
                "Nearest Soil Map Cell",
                f"{apna_distance:.2f} km",
            )

            st.metric(
                "Texture Classification",
                texture,
            )

            apna_properties = {
                "EC": "Electrical Conductivity",
                "pH": "pH",
                "OM": "Organic Matter",
                "P": "Phosphorus",
                "K": "Potassium",
            }

            apna_rows = []

            for key, label in apna_properties.items():

                apna_rows.append({
                    "Property": label,
                    "Map Information": format_apna_property(
                        apna_row,
                        key,
                    ),
                })

            st.dataframe(
                pd.DataFrame(apna_rows),
                hide_index=True,
                width="stretch",
            )

        else:

            st.info(
                "No APNA soil-map cell found within "
                f"{APNA_SEARCH_LIMIT_KM:.0f} km."
            )

    # =====================================================
    # TAB 3 — CROP COMPATIBILITY
    # =====================================================

    with tab_crops:

        st.markdown(
            '<div class="section-heading">Crop pH Compatibility Analysis</div>',
            unsafe_allow_html=True,
        )

        if ph_value is not None:

            st.metric(
                "Analyzed Soil pH",
                f"{ph_value:.2f}",
            )

            st.write(
                "Comparison of estimated soil pH against "
                "broad crop pH suitability ranges."
            )

            crop_df = pd.DataFrame(crop_results)

            st.dataframe(
                crop_df,
                hide_index=True,
                width="stretch",
            )

            compatible_count = sum(
                item["Compatibility"] == "Compatible"
                for item in crop_results
            )

            near_count = sum(
                item["Compatibility"] == "Near boundary"
                for item in crop_results
            )

            outside_count = sum(
                item["Compatibility"] == "Outside range"
                for item in crop_results
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Compatible",
                compatible_count,
            )

            c2.metric(
                "Near Boundary",
                near_count,
            )

            c3.metric(
                "Outside Range",
                outside_count,
            )

        else:

            st.info(
                "Soil pH is unavailable for crop compatibility analysis."
            )

        st.caption(
            "Compatibility is based on pH only. "
            "It does not represent a complete agronomic crop recommendation."
        )

    # =====================================================
    # TAB 4 — METHODOLOGY
    # =====================================================

    with tab_method:

        st.markdown(
            '<div class="section-heading">System Methodology</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            **1. Geographic Input**

            The system receives latitude and longitude
            coordinates for a selected location.

            **2. Spatial Data Retrieval**

            The coordinate is matched with the nearest
            available APNA Farm grid cell and SoilGrids
            raster pixels.

            **3. Soil Property Extraction**

            Eight topsoil properties are retrieved:
            pH, organic carbon, nitrogen, clay, sand,
            silt, cation exchange capacity and bulk density.

            **4. Soil Condition Assessment**

            Rule-based indicators examine selected
            soil properties and identify potential
            conditions requiring attention.

            **5. Crop Compatibility**

            Estimated soil pH is compared with broad
            crop-specific pH ranges.
            """
        )

        st.divider()

        st.markdown("**Spatial data sources**")

        st.markdown(
            """
            - APNA Farm digitized Punjab soil maps.
            - SoilGrids modelled soil-property rasters.
            - Topsoil profile: 0–5 cm.
            - Raster scale conversion applied during extraction.
            """
        )

        st.markdown("**System components**")

        st.markdown(
            """
            - Python
            - Streamlit
            - Pandas and NumPy
            - Rasterio
            - Geospatial coordinate transformation
            """
        )

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        """
        <div class="footer">
        SoilSenseAI | Punjab Soil Intelligence System<br>
        Geospatial Soil Analysis & Crop Compatibility Platform
        </div>
        """,
        unsafe_allow_html=True,
    )
