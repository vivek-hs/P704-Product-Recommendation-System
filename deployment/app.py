import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from joblib import load
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"


@st.cache_resource
def load_artifacts():
    """Load and validate saved recommendation and onboarding artifacts once per process."""
    bundle = load(ARTIFACT_DIR / "approach_a_recommendations.joblib")
    seen_items = load_npz(ARTIFACT_DIR / "seen_items.npz").tocsr()
    product_metadata = load(ARTIFACT_DIR / "product_metadata.joblib")
    behavioural_pipeline = load(ARTIFACT_DIR / "behavioural_kmeans_pipeline.joblib")

    expected_features = [
        "log_rating_count", "mean_rating", "rating_std", "positive_ratio", "negative_ratio"
    ]
    if behavioural_pipeline.get("artifact_version") != 1:
        raise ValueError("Unsupported behavioural pipeline artifact version.")
    if behavioural_pipeline.get("training_only") is not True:
        raise ValueError("The behavioural pipeline must be fitted on training-only data.")
    if behavioural_pipeline.get("selected_k") != 4:
        raise ValueError("The saved behavioural pipeline must use k=4.")
    if behavioural_pipeline.get("feature_order") != expected_features:
        raise ValueError("The behavioural feature order does not match the supported schema.")
    if behavioural_pipeline.get("minimum_onboarding_ratings") != 5:
        raise ValueError("The onboarding artifact must require at least five ratings.")
    if behavioural_pipeline.get("positive_rating_min") != 4:
        raise ValueError("The positive-rating definition is missing or unsupported.")
    if behavioural_pipeline.get("negative_rating_max") != 2:
        raise ValueError("The negative-rating definition is missing or unsupported.")
    if behavioural_pipeline.get("rating_std_ddof") != 1:
        raise ValueError("The sample-standard-deviation definition is missing or unsupported.")
    if behavioural_pipeline.get("rating_std_fill_value") != 0.0:
        raise ValueError("The rating-standard-deviation fill value is unsupported.")
    if behavioural_pipeline.get("count_transform") != "np.log1p":
        raise ValueError("The activity-count transform is unsupported.")
    if behavioural_pipeline.get("sklearn_version") != "1.7.2":
        raise ValueError("This app expects the behavioural artifact created with scikit-learn 1.7.2.")
    if sklearn.__version__ != behavioural_pipeline["sklearn_version"]:
        raise ValueError(
            f"Installed scikit-learn is {sklearn.__version__}; "
            f"the artifact requires {behavioural_pipeline['sklearn_version']}."
        )

    scaler = behavioural_pipeline.get("scaler")
    kmeans_model = behavioural_pipeline.get("kmeans_model")
    if not isinstance(scaler, StandardScaler):
        raise ValueError("The saved behavioural scaler is missing or has an unexpected type.")
    if not isinstance(kmeans_model, KMeans):
        raise ValueError("The saved behavioural KMeans model is missing or has an unexpected type.")
    check_is_fitted(scaler)
    check_is_fitted(kmeans_model)
    if int(kmeans_model.n_clusters) != int(bundle.get("n_clusters", -1)):
        raise ValueError("The behavioural KMeans cluster count does not match the recommendation bundle.")
    if int(kmeans_model.n_clusters) != int(behavioural_pipeline["selected_k"]):
        raise ValueError("The fitted KMeans model does not match the selected cluster count.")

    return bundle, seen_items, product_metadata, behavioural_pipeline


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


def validate_onboarding_ratings(edited_ratings, product_catalog, minimum_ratings):
    """Validate edited Product ID/rating rows and return clean pairs plus messages."""
    errors = []
    ratings = []
    seen_ids = set()
    duplicate_ids = set()

    for row_number, (_, row) in enumerate(edited_ratings.iterrows(), start=1):
        raw_product_id = row["Product ID"]
        raw_rating = row["Rating"]
        product_id_empty = pd.isna(raw_product_id) or not str(raw_product_id).strip()
        rating_empty = pd.isna(raw_rating) or not str(raw_rating).strip()

        if product_id_empty and rating_empty:
            continue
        if product_id_empty:
            errors.append(f"Row {row_number}: enter a Product ID for this rating.")
            continue

        # Keep the ID as trimmed text so leading zeroes remain part of the identifier.
        product_id = str(raw_product_id).strip()
        if product_id in seen_ids:
            duplicate_ids.add(product_id)
        seen_ids.add(product_id)

        row_valid = True
        if product_id not in product_catalog:
            errors.append(f"Row {row_number}: Product ID `{product_id}` is not in the saved product catalog.")
            row_valid = False

        if rating_empty:
            errors.append(f"Row {row_number}: enter a rating from 1 to 5.")
            row_valid = False
        else:
            try:
                numeric_rating = float(raw_rating)
            except (TypeError, ValueError):
                numeric_rating = np.nan

            if not np.isfinite(numeric_rating) or not numeric_rating.is_integer():
                errors.append(f"Row {row_number}: rating must be a whole number from 1 to 5.")
                row_valid = False
            elif numeric_rating < 1 or numeric_rating > 5:
                errors.append(f"Row {row_number}: rating must be between 1 and 5.")
                row_valid = False

        if row_valid:
            ratings.append((product_id, int(numeric_rating)))

    if duplicate_ids:
        duplicates = ", ".join(sorted(duplicate_ids))
        errors.append(f"Product IDs must be unique. Duplicates: {duplicates}.")
    if len(ratings) < minimum_ratings:
        errors.append(f"Enter at least {minimum_ratings} complete, valid product ratings.")
    if len({product_id for product_id, _ in ratings}) < minimum_ratings:
        errors.append(f"At least {minimum_ratings} unique Product IDs are required.")

    return ratings, errors


def make_onboarding_feature_frame(ratings, behavioural_pipeline):
    """Build the exact one-row behavioural feature frame expected by the saved scaler."""
    rating_values = pd.Series([rating for _, rating in ratings], dtype="float64")
    rating_count = len(rating_values)
    rating_std = rating_values.std(ddof=behavioural_pipeline["rating_std_ddof"])
    if pd.isna(rating_std):
        rating_std = behavioural_pipeline["rating_std_fill_value"]

    feature_values = {
        "rating_count": rating_count,
        "log_rating_count": float(np.log1p(rating_count)),
        "mean_rating": float(rating_values.mean()),
        "rating_std": float(rating_std),
        "positive_ratio": float(
            (rating_values >= behavioural_pipeline["positive_rating_min"]).mean()
        ),
        "negative_ratio": float(
            (rating_values <= behavioural_pipeline["negative_rating_max"]).mean()
        ),
    }
    feature_order = behavioural_pipeline["feature_order"]
    feature_frame = pd.DataFrame(
        [[feature_values[name] for name in feature_order]],
        columns=feature_order,
    )
    summary = {
        "rating_count": rating_count,
        "mean_rating": feature_values["mean_rating"],
        "rating_std": feature_values["rating_std"],
        "positive_ratio": feature_values["positive_ratio"],
        "negative_ratio": feature_values["negative_ratio"],
    }
    return feature_frame, summary


def get_cluster_recommendations(cluster_id, number, bundle, product_metadata, excluded_product_ids):
    """Return ranked unseen products from one saved cluster, without a popularity fallback."""
    candidate_products = bundle["cluster_ranked_products"].get(cluster_id, [])
    stats_by_product = product_metadata["cluster_product_stats"][cluster_id]
    excluded_product_ids = set(excluded_product_ids)
    recommendations = []

    for product_id in candidate_products:
        if product_id in excluded_product_ids:
            continue
        recommendations.append(product_id)
        if len(recommendations) >= number:
            break

    return recommendations, len(candidate_products), stats_by_product


def parse_manual_seen_ids(value):
    """Trim comma-separated product IDs and remove duplicates without numeric conversion."""
    product_ids = []
    seen_ids = set()

    for item in value.split(","):
        product_id = item.strip()
        if product_id and product_id not in seen_ids:
            product_ids.append(product_id)
            seen_ids.add(product_id)

    return product_ids


def count_training_candidates_excluded(user_id, cluster_id, bundle, seen_items):
    """Count products in the saved cluster list already present in the user's history."""
    user_id = str(user_id).strip()
    user_row = int(np.searchsorted(bundle["user_ids"], user_id))
    if user_row >= len(bundle["user_ids"]) or bundle["user_ids"][user_row] != user_id:
        return 0

    first_seen = seen_items.indptr[user_row]
    last_seen = seen_items.indptr[user_row + 1]
    seen_columns = set(seen_items.indices[first_seen:last_seen])
    candidate_products = bundle["cluster_ranked_products"].get(cluster_id, [])
    product_ids = bundle["product_ids"]
    excluded_count = 0

    for product_id in candidate_products:
        product_column = int(np.searchsorted(product_ids, product_id))
        if product_column < len(product_ids) and product_ids[product_column] == product_id:
            if product_column in seen_columns:
                excluded_count += 1

    return excluded_count


def make_download_csv(rows):
    """Format the displayed recommendation rows as a downloadable CSV."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def display_profile_label(profile):
    return profile["label"].replace("-", " ").capitalize()


st.set_page_config(page_title="Product Recommendation System", layout="wide")
st.title("Product Recommendation System")
st.write(
    "Personalized recommendations from behavioural user clusters, with onboarding for new users "
    "and Global Popularity as a fallback when no usable profile is available."
)
st.caption("Behavioural KMeans · k=4 · Streamlit")

try:
    bundle, seen_items, product_metadata, behavioural_pipeline = load_artifacts()
except Exception as error:
    st.error(f"Could not load deployment artifacts: {error}")
    st.stop()

if product_metadata.get("training_only") is not True:
    st.error("Product statistics are not marked as training-only in the saved artifact.")
    st.stop()

product_catalog = {str(product_id) for product_id in bundle["product_ids"]}

demo_users = bundle.get("demo_users", [])
demo_user_ids = []
for demo_user in demo_users:
    demo_user_id = str(demo_user.get("user_id", "")).strip()
    if demo_user_id and demo_user_id not in demo_user_ids:
        demo_user_ids.append(demo_user_id)

placeholder = "Select a demonstration user"
recommendations_tab, new_user_tab, insights_tab, about_tab = st.tabs(
    ["Recommendations", "New User", "Model Insights", "About"]
)

with recommendations_tab:
    st.subheader("Recommend for an existing user")
    with st.form("recommendation_inputs"):
        user_inputs, ranking_inputs = st.columns(2)
        with user_inputs:
            selected_demo = st.selectbox(
                "Demonstration user",
                [placeholder] + demo_user_ids,
                help="Choose one of the saved example users.",
            )
            typed_user_id = st.text_input(
                "Manual User ID (optional)",
                help="Enter an ID exactly as stored. Manual input overrides the demonstration-user selection.",
            )
        with ranking_inputs:
            number = st.slider(
                "Number of recommendations",
                min_value=1,
                max_value=20,
                value=10,
            )
            with st.expander("Optional exclusions"):
                extra_seen_text = st.text_input(
                    "Already-rated Product IDs",
                    help="Enter comma-separated Product IDs to exclude. Spaces are trimmed, duplicates are ignored, and IDs remain text.",
                )
        submitted = st.form_submit_button("Get Recommendations", type="primary")

    if submitted:
        manual_user_id = typed_user_id.strip()

        if manual_user_id:
            user_id = manual_user_id
        elif selected_demo == placeholder:
            user_id = ""
        elif selected_demo not in demo_user_ids:
            user_id = ""
            st.error("That demonstration user is not available in the current saved artifact.")
        else:
            user_id = selected_demo

        if not user_id:
            if selected_demo == placeholder and not manual_user_id:
                st.warning("Enter a User ID or select a demonstration user.")
        else:
            extra_seen_ids = parse_manual_seen_ids(extra_seen_text)
            known_user, cluster_id, recommendations = get_recommendations(
                user_id, number, bundle, seen_items, extra_seen_ids
            )

            st.subheader("Recommendation results")
            st.write(f"**User ID:** `{user_id}`")

            if known_user:
                profile = bundle["cluster_profiles"].get(cluster_id)
                if profile is None:
                    st.error("The saved cluster profile is unavailable for this user.")
                    st.stop()

                training_excluded_count = count_training_candidates_excluded(
                    user_id, cluster_id, bundle, seen_items
                )
                summary = st.columns([1, 1, 2])
                summary[0].metric("User type", "Known user")
                summary[1].metric("Cluster", f"Cluster {cluster_id}")
                summary[2].metric("Segment", display_profile_label(profile))
                if training_excluded_count == 1:
                    exclusion_message = (
                        "1 ranked candidate product was excluded because it was already present "
                        "in this user's training history."
                    )
                else:
                    exclusion_message = (
                        f"{training_excluded_count} ranked candidate products were excluded because "
                        "they were already present in this user's training history."
                    )
                st.caption(exclusion_message)
                stats_by_product = product_metadata["cluster_product_stats"][cluster_id]
                source = f"Cluster {cluster_id}"
            else:
                summary = st.columns(2)
                summary[0].metric("User type", "Unsupported user")
                summary[1].metric("Recommendation source", "Global Popularity")
                st.info(
                    "No saved behavioural cluster or training history is available for this user. "
                    "Global Popularity is used as the fallback; manually supplied Product IDs are still excluded."
                )
                stats_by_product = product_metadata["global_product_stats"]
                source = "Global Popularity"

            with st.expander("Why these recommendations?"):
                if known_user:
                    st.markdown(
                        "- Products are ranked using this user's saved behavioural cluster and its training-only ranking.\n"
                        "- Training-history products and optional exclusions are skipped.\n"
                        "- The highest-ranked unseen products are returned."
                    )
                else:
                    st.markdown(
                        "- No saved behavioural cluster is available, so the Global Popularity ranking is used.\n"
                        "- Optional manually supplied Product IDs are excluded."
                    )

            st.subheader("Recommended products")
            if recommendations:
                table_rows = []
                for rank, product_id in enumerate(recommendations, start=1):
                    product_stats = stats_by_product[product_id]
                    table_rows.append({
                        "Rank": rank,
                        "Product ID": str(product_id),
                        "Avg Rating": product_stats["avg_rating"],
                        "Rating Count": product_stats["rating_count"],
                        "Positive %": product_stats["positive_percent"],
                        "Source": source,
                    })

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
                    f"Returned {len(recommendations)} of {number} requested. "
                    "Historical statistics are training-only and are not predicted ratings."
                )
                st.download_button(
                    "Download recommendations as CSV",
                    data=make_download_csv(table_rows),
                    file_name="p704_recommendations.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No unseen products remain in the saved ranking after the selected exclusions.")

with new_user_tab:
    st.subheader("Build an initial behavioural profile")
    st.write(
        "Rate the five sample Product IDs below from 1 to 5. Because the source dataset contains only Product IDs, "
        "this is a behavioural onboarding demonstration rather than a semantic product-preference survey."
    )
    st.write(
        "Your ratings are converted into five behavioural features and passed through the saved scaler and "
        "KMeans model. The model is not retrained."
    )

    global_stats = product_metadata["global_product_stats"]
    suggested_rows = []
    starter_product_ids = []
    for product_id in bundle["global_ranked_products"]:
        product_stats = global_stats.get(product_id)
        if product_stats is None:
            continue
        product_id_text = str(product_id)
        if len(starter_product_ids) < 5:
            starter_product_ids.append(product_id_text)
        suggested_rows.append({
            "Product ID": product_id_text,
            "Avg Rating": product_stats["avg_rating"],
            "Rating Count": product_stats["rating_count"],
            "Positive %": product_stats["positive_percent"],
        })
        if len(suggested_rows) == 25:
            break

    starter_product_ids.extend([""] * (5 - len(starter_product_ids)))
    entered_ratings = []
    with st.form("new_user_onboarding"):
        for row_number, product_id in enumerate(starter_product_ids):
            product_column, rating_column = st.columns([3, 1])
            product_column.markdown(f"**Product ID:** `{product_id}`")
            selected_rating = rating_column.selectbox(
                "Your Rating",
                options=[None, 1, 2, 3, 4, 5],
                format_func=lambda value: "Select rating" if value is None else str(value),
                key=f"new_user_rating_{row_number}",
            )
            entered_ratings.append(selected_rating)

        new_user_number = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=10,
            key="new_user_top_n",
        )
        build_recommendations = st.form_submit_button(
            "Build My Recommendations", type="primary"
        )

    with st.expander("Browse more sample Product IDs"):
        st.dataframe(
            suggested_rows,
            column_config={
                "Avg Rating": st.column_config.NumberColumn(format="%.2f"),
                "Rating Count": st.column_config.NumberColumn(format="%d"),
                "Positive %": st.column_config.NumberColumn(format="%.1f%%"),
            },
            hide_index=True,
            width="stretch",
        )
        st.caption("These are historical training statistics, not predicted ratings.")

    if build_recommendations:
        rating_values = [pd.NA if rating is None else rating for rating in entered_ratings]
        onboarding_input = pd.DataFrame({
            "Product ID": pd.Series(starter_product_ids, dtype="string"),
            "Rating": pd.Series(rating_values, dtype="Int64"),
        })
        onboarding_ratings, validation_errors = validate_onboarding_ratings(
            onboarding_input,
            product_catalog,
            behavioural_pipeline["minimum_onboarding_ratings"],
        )
        if validation_errors:
            error_list = "\n".join(f"- {message}" for message in dict.fromkeys(validation_errors))
            st.error(f"Please correct the following onboarding entries:\n\n{error_list}")
        else:
            feature_frame, onboarding_profile = make_onboarding_feature_frame(
                onboarding_ratings, behavioural_pipeline
            )
            scaled_profile = behavioural_pipeline["scaler"].transform(feature_frame)
            cluster_id = int(behavioural_pipeline["kmeans_model"].predict(scaled_profile)[0])
            cluster_profile = bundle["cluster_profiles"].get(cluster_id)

            if cluster_profile is None:
                st.error("The predicted cluster profile is not available in the saved recommendation bundle.")
            else:
                summary = st.columns([1, 1, 2])
                summary[0].metric("Ratings provided", onboarding_profile["rating_count"])
                summary[1].metric("Initial cluster", f"Cluster {cluster_id}")
                summary[2].metric("User segment", display_profile_label(cluster_profile))
                st.caption(
                    f"Mean rating: {onboarding_profile['mean_rating']:.2f}  ·  "
                    f"Positive ratings: {100 * onboarding_profile['positive_ratio']:.1f}%  ·  "
                    f"Negative ratings: {100 * onboarding_profile['negative_ratio']:.1f}%"
                )

                with st.expander("Why this cluster?"):
                    st.write(
                        "The initial assignment uses activity, average rating, rating spread, positive share, "
                        "and negative share. It does not infer product meaning or demographics, and may change "
                        "as more ratings become available."
                    )
                with st.expander("Rating spread"):
                    st.write(f"Sample rating standard deviation: {onboarding_profile['rating_std']:.3f}")

                rated_product_ids = {product_id for product_id, _ in onboarding_ratings}
                recommendations, candidate_count, stats_by_product = get_cluster_recommendations(
                    cluster_id,
                    new_user_number,
                    bundle,
                    product_metadata,
                    rated_product_ids,
                )

                st.subheader(f"Recommendations from initial Cluster {cluster_id}")
                if recommendations:
                    table_rows = []
                    for rank, product_id in enumerate(recommendations, start=1):
                        product_stats = stats_by_product[product_id]
                        table_rows.append({
                            "Rank": rank,
                            "Product ID": str(product_id),
                            "Avg Rating": product_stats["avg_rating"],
                            "Rating Count": product_stats["rating_count"],
                            "Positive %": product_stats["positive_percent"],
                            "Source": f"Predicted Cluster {cluster_id}",
                        })

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
                        f"Returned {len(recommendations)} of {new_user_number} requested. "
                        "Historical statistics are training-only and are not predicted ratings."
                    )

                if len(recommendations) < new_user_number:
                    st.warning(
                        f"Returned {len(recommendations)} of {new_user_number} requested; "
                        f"the saved cluster ranking had {candidate_count} products before exclusions. "
                        "This onboarding path does not switch to Global Popularity."
                    )

with insights_tab:
    st.subheader("Behavioural user clusters")
    st.write(
        f"Approach A assigns modelled users to {bundle['n_clusters']} saved KMeans clusters. "
        "The profiles summarize training-user activity and rating preferences."
    )

    st.subheader("Cluster population")
    cluster_share = pd.DataFrame(
        {
            "Cluster": [f"Cluster {cluster_id}" for cluster_id in sorted(bundle["cluster_profiles"])],
            "User share (%)": [
                bundle["cluster_profiles"][cluster_id]["share_percent"]
                for cluster_id in sorted(bundle["cluster_profiles"])
            ],
        }
    )
    st.bar_chart(
        cluster_share,
        x="Cluster",
        y="User share (%)",
        horizontal=True,
        height=280,
    )

    profile_rows = []
    for cluster_id in sorted(bundle["cluster_profiles"]):
        profile = bundle["cluster_profiles"][cluster_id]
        profile_rows.append({
            "Cluster": f"Cluster {cluster_id}",
            "User segment": display_profile_label(profile),
            "Users": profile["users"],
            "User share": profile["share_percent"],
            "Mean rating": profile["mean_rating"],
            "Positive ratings": 100 * profile["positive_ratio"],
            "Negative ratings": 100 * profile["negative_ratio"],
        })

    st.dataframe(
        profile_rows,
        column_config={
            "Users": st.column_config.NumberColumn(format="%d"),
            "User share": st.column_config.NumberColumn(format="%.2f%%"),
            "Mean rating": st.column_config.NumberColumn(format="%.2f"),
            "Positive ratings": st.column_config.NumberColumn(format="%.1f%%"),
            "Negative ratings": st.column_config.NumberColumn(format="%.1f%%"),
        },
        hide_index=True,
        width="stretch",
    )
    st.caption("Cluster summaries are read from the saved training-only recommendation bundle.")

with about_tab:
    st.subheader("How recommendations work")
    st.markdown(
        "- **Existing modelled users:** use their saved cluster ranking, excluding products rated in training.\n"
        "- **New users with at least five ratings:** receive an initial behavioural cluster and recommendations from its saved ranking.\n"
        "- **Users without usable behavioural history:** receive the Global Popularity fallback."
    )

    st.subheader("Technical notes")
    st.markdown(
        "- There is no runtime retraining and the app does not load ratings.csv.\n"
        "- The fitted scaler and KMeans model are loaded from saved artifacts.\n"
        "- Displayed statistics are historical and training-only.\n"
        "- Product names are unavailable because the source dataset contains Product IDs only."
    )
