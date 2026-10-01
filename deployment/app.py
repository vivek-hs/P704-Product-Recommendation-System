from pathlib import Path

import numpy as np
import streamlit as st
from joblib import load
from scipy.sparse import load_npz


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"


@st.cache_resource
def load_artifacts():
    bundle = load(ARTIFACT_DIR / "approach_a_recommendations.joblib")
    seen_items = load_npz(ARTIFACT_DIR / "seen_items.npz").tocsr()
    product_metadata = load(ARTIFACT_DIR / "product_metadata.joblib")
    return bundle, seen_items, product_metadata


def get_recommendations(user_id, number, bundle, seen_items, extra_seen_ids):
    user_id = str(user_id).strip()
    user_ids = bundle["user_ids"]
    user_row = int(np.searchsorted(user_ids, user_id))
    known_user = user_row < len(user_ids) and user_ids[user_row] == user_id

    if known_user:
        cluster_id = int(bundle["cluster_labels"][user_row])
        first_seen = seen_items.indptr[user_row]
        last_seen = seen_items.indptr[user_row + 1]
        seen_columns = set(seen_items.indices[first_seen:last_seen])
        candidate_products = bundle["cluster_ranked_products"].get(cluster_id, [])
    else:
        cluster_id = None
        seen_columns = set()
        candidate_products = bundle["global_ranked_products"]

    extra_seen_ids = set(extra_seen_ids)
    product_ids = bundle["product_ids"]
    recommendations = []

    for product_id in candidate_products:
        if product_id in extra_seen_ids:
            continue

        product_column = int(np.searchsorted(product_ids, product_id))
        if product_column < len(product_ids) and product_ids[product_column] == product_id:
            if product_column in seen_columns:
                continue

        recommendations.append(product_id)
        if len(recommendations) >= number:
            break

    return known_user, cluster_id, recommendations


st.set_page_config(page_title="P704 Product Recommendations", page_icon="🛍️")
st.title("P704 Product Recommendations")
st.caption("Approach A · Behavioural User Clustering using KMeans (k=4)")

try:
    bundle, seen_items, product_metadata = load_artifacts()
except Exception as error:
    st.error(f"Could not load deployment artifacts: {error}")
    st.stop()

demo_users = bundle["demo_users"]
demo_options = ["Enter a userId"] + [row["user_id"] for row in demo_users]
selected_demo = st.selectbox("Select a demonstration user", demo_options)
typed_user_id = st.text_input("Or enter a userId")
st.caption("Manual userId input overrides the demonstration-user dropdown.")
user_id = typed_user_id.strip()
if not user_id and selected_demo != "Enter a userId":
    user_id = selected_demo

number = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)
extra_seen_text = st.text_input(
    "Additional already-rated productIds (optional, comma-separated)",
    help="Useful for users outside the modeled population. These IDs are excluded from the results.",
)
extra_seen_ids = [value.strip() for value in extra_seen_text.split(",") if value.strip()]

if user_id:
    known_user, cluster_id, recommendations = get_recommendations(
        user_id, number, bundle, seen_items, extra_seen_ids
    )

    st.subheader("Recommendation results")
    st.write(f"**userId:** `{user_id}`")

    if known_user:
        profile = bundle["cluster_profiles"][cluster_id]
        cluster_value = f"Cluster {cluster_id}"
        if "higher-activity" in profile["label"]:
            segment_value = "Higher Activity"
        else:
            segment_value = profile["label"].split(" / ")[-1].title()
        full_profile = profile["label"].replace("-", " ").capitalize()
        source = cluster_value
        stats_by_product = product_metadata["cluster_product_stats"][cluster_id]
        st.caption("Products already rated in the training data are excluded.")
    else:
        cluster_value = "Global Popularity Fallback"
        segment_value = "Not modelled"
        source = "Global Popularity"
        stats_by_product = product_metadata["global_product_stats"]
        st.info(
            "This user is not represented in the clustered modeling population. "
            "Showing the Global Popularity fallback. "
            "The app has no saved rating history for this user, so enter any known "
            "already-rated productIds above to exclude them."
        )

    metric_columns = st.columns(3)
    metric_columns[0].metric("Assigned Cluster", cluster_value)
    metric_columns[1].metric("User Segment", segment_value)
    metric_columns[2].metric("Number of Recommendations", len(recommendations))

    if known_user:
        st.caption(f"Cluster profile: {full_profile}")

    if recommendations:
        table_rows = []
        for rank, product_id in enumerate(recommendations, start=1):
            product_stats = stats_by_product[product_id]
            table_rows.append({
                "Rank": rank,
                "Product ID": product_id,
                "Avg Rating": product_stats["avg_rating"],
                "Rating Count": product_stats["rating_count"],
                "Positive %": product_stats["positive_percent"],
                "Source": source,
            })

        st.write("**Recommended products**")
        st.dataframe(
            table_rows,
            column_config={
                "Avg Rating": st.column_config.NumberColumn(format="%.2f"),
                "Rating Count": st.column_config.NumberColumn(format="%d"),
                "Positive %": st.column_config.NumberColumn(format="%.1f%%"),
            },
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Average rating, rating count and positive percentage are historical "
            "training statistics for the displayed source, not predicted ratings."
        )
    else:
        st.warning("No unseen products were available in the saved recommendation list.")
else:
    st.info("Select a demonstration user or enter a userId to get recommendations.")

with st.expander("About this demo"):
    st.write(
        "The app uses precomputed clusters and training-only rankings. "
        "It does not retrain the model or require ratings.csv. Product names are "
        "not available in the source data, so results are shown as productIds."
    )
