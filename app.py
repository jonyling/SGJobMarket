import io
import json
import re
import zipfile
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SG Job Market Intelligence",
    page_icon="SG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Singapore Job Market Intelligence")
st.markdown("Data-driven insights for headhunters and job seekers.")
st.markdown("### Data-Driven Insights for Headhunters & Job Seekers")

# -----------------------------------------------------------------------------
# DATA LOADING AND PREPROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_prepare(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Drop rows with missing critical data
    critical_columns = [
        "categories",
        "employmentTypes",
        "title",
        "salary_minimum",
        "salary_maximum",
        "postedCompany_name",
        "positionLevels",
        "metadata_totalNumberOfView",
        "minimumYearsExperience",
    ]
    df = df.dropna(subset=critical_columns)

    # Ruthless filtering
    df = df[df["minimumYearsExperience"] <= 10]
    df = df[df["salary_maximum"] <= 30000]
    df = df[df["salary_minimum"] >= 1000]

    # Average salary
    df["average_salary"] = (df["salary_minimum"] + df["salary_maximum"]) / 2

    # Parse categories into list
    def parse_categories(cats_str):
        if pd.isna(cats_str) or str(cats_str).strip() in ["", "nan"]:
            return []
        try:
            cats = json.loads(str(cats_str).replace("''", '"').replace('""', '"'))
            return sorted([item["category"] for item in cats if "category" in item])
        except Exception:
            return []

    df["category_list"] = df["categories"].apply(parse_categories)

    # Primary category
    df["primary_category"] = df["category_list"].apply(lambda x: x[0] if len(x) > 0 else "Unknown")

    # Clean titles
    def clean_title(title):
        if pd.isna(title):
            return ""
        title = str(title).lower()
        title = re.sub(r"\$?\s*\d{3,6}(?:-\d{3,6})?\s*(?:k|sgd|monthly|pm)?", " ", title)
        title = re.sub(r"\b(singapore|west|east|north|central|clementi|tuas|jurong|chang i).*?\b", " ", title)
        title = re.sub(r"\b(up to|urgent|immediate|asap|attractive|entry level|no exp|fresh grad|bonus)\b", " ", title)
        title = re.sub(r"\|.*", " ", title)
        title = re.sub(r"\([^)]*\)", " ", title)
        title = re.sub(r"[^a-z0-9\s]", " ", title)
        title = re.sub(r"\s+", " ", title).strip()
        return title

    df["clean_title"] = df["title"].apply(clean_title)

    # Key role extraction
    priority_roles = [
        "business development manager",
        "customer service officer",
        "business development executive",
        "customer service executive",
        "chef de partie",
        "senior software engineer",
        "patient service associate",
        "digital marketing executive",
        "full stack developer",
        "human resource executive",
        "resident technical officer",
        "assistant restaurant manager",
        "senior accounts executive",
        "senior hr executive",
        "hr admin executive",
        "desktop support engineer",
        "senior quantity surveyor",
        "senior staff nurse",
        "mechanical design engineer",
        "field service engineer",
        "hr recruitment consultant",
        "senior project engineer",
        "sales support executive",
        "it support engineer",
        "assistant project manager",
        "software engineer",
        "project manager",
        "sales executive",
        "admin executive",
        "accounts executive",
        "project engineer",
        "quantity surveyor",
        "sales manager",
        "marketing executive",
        "hr executive",
        "business analyst",
        "staff nurse",
        "service engineer",
        "design engineer",
        "account manager",
    ]

    def extract_key_role(clean_title_value):
        if not clean_title_value:
            return "unknown"
        padded = f" {clean_title_value} "
        for role in priority_roles:
            if f" {role} " in padded:
                return role
        words = padded.split()
        return " ".join(words[:3]) if len(words) >= 3 else " ".join(words)

    df["key_role"] = df["clean_title"].apply(extract_key_role)

    # Time parsing
    df["original_post_date"] = pd.to_datetime(df["metadata_originalPostingDate"], errors="coerce")
    df = df[df["original_post_date"].notna()].copy()
    df["year_month"] = df["original_post_date"].dt.to_period("M").astype(str) # type: ignore
    df["year"] = df["original_post_date"].dt.year # type: ignore
    df["quarter"] = df["original_post_date"].dt.to_period("Q").astype(str) # type: ignore

    # Engagement metrics
    df["metadata_repostCount"] = df["metadata_repostCount"].fillna(0)
    df["metadata_totalNumberJobApplication"] = df["metadata_totalNumberJobApplication"].fillna(0)
    df["application_rate"] = (
        df["metadata_totalNumberJobApplication"] / df["metadata_totalNumberOfView"] * 100
    ).fillna(0)

    return df


DATA_PATH = "../data/SGJobData_part17.csv"

try:
    df = load_and_prepare(DATA_PATH)
except FileNotFoundError:
    st.error(f"Data file not found at {DATA_PATH}. Place the CSV in ../data/.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

all_categories = sorted(df["primary_category"].unique())
selected_categories = st.sidebar.multiselect(
    "Industries", all_categories, default=all_categories[:5]
)

min_sal = int(df["salary_minimum"].min())
max_sal = int(df["salary_maximum"].max())
salary_range = st.sidebar.slider("Monthly Salary (SGD)", min_sal, max_sal, (min_sal, 15000))

exp_range = st.sidebar.slider("Years of Experience", 0, 10, (0, 10))

min_year, max_year = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider("Posting Year", min_year, max_year, (min_year, max_year))

mask = (
    df["primary_category"].isin(selected_categories)
    & (df["average_salary"] >= salary_range[0])
    & (df["average_salary"] <= salary_range[1])
    & (df["minimumYearsExperience"] >= exp_range[0])
    & (df["minimumYearsExperience"] <= exp_range[1])
    & (df["year"] >= year_range[0])
    & (df["year"] <= year_range[1])
)

filtered = df[mask].copy()

# -----------------------------------------------------------------------------
# KPI ROW
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Jobs", f"{len(filtered):,}")
with col2:
    st.metric("Avg Salary", f"${int(filtered['average_salary'].mean()):,}")
with col3:
    st.metric("Median Experience", f"{int(filtered['minimumYearsExperience'].median())} yrs")
with col4:
    top_sector = filtered["primary_category"].mode()[0] if not filtered.empty else "N/A"
    st.metric("Top Sector", top_sector)

st.markdown("---")

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Market Overview",
        "Category x Key Role",
        "Time Series",
        "QoQ Growth",
        "Compensation Intelligence",
        "Headhunter Matrix",
        "Quick vs Hard to Fill",
        "Job Seeker Explorer",
    ]
)

with tab1:
    st.subheader("Market Demand by Industry")

    cat_counts = filtered["primary_category"].value_counts().reset_index()
    cat_counts.columns = ["Industry", "Job Count"]

    fig_cat = px.bar(
        cat_counts.head(15),
        x="Job Count",
        y="Industry",
        orientation="h",
        color="Job Count",
        color_continuous_scale="viridis",
        title="Top Industries by Posting Volume",
    )
    fig_cat.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_cat, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        emp_counts = filtered["employmentTypes"].value_counts().reset_index()
        emp_counts.columns = ["Type", "Count"]
        fig_pie = px.pie(emp_counts, values="Count", names="Type", title="Employment Types")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        exp_counts = filtered["minimumYearsExperience"].value_counts().sort_index().reset_index()
        exp_counts.columns = ["Years Exp", "Count"]
        fig_exp = px.bar(exp_counts, x="Years Exp", y="Count", title="Demand by Experience")
        st.plotly_chart(fig_exp, use_container_width=True)

with tab2:
    st.subheader("Crosslink: Category and Key Role")

    df_exploded = filtered.explode("category_list")
    df_exploded = df_exploded[df_exploded["category_list"].notna()]
    df_exploded = df_exploded[df_exploded["category_list"].str.strip() != ""]

    top_categories = df_exploded["category_list"].value_counts().head(12).index.tolist()
    top_roles = df_exploded["key_role"].value_counts().head(15).index.tolist()

    matrix = (
        df_exploded[df_exploded["category_list"].isin(top_categories) & df_exploded["key_role"].isin(top_roles)]
        .groupby(["category_list", "key_role"])
        .size()
        .reset_index(name="postings")
    )

    pivot = matrix.pivot(index="category_list", columns="key_role", values="postings").fillna(0)

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Viridis",
            colorbar_title="Postings",
        )
    )
    fig_heat.update_layout(
        title="Category x Key Role Heatmap (Top Categories/Roles)",
        xaxis_title="Key Role",
        yaxis_title="Category",
        height=600,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("Top key roles per category")
    for cat in top_categories[:6]:
        subset = df_exploded[df_exploded["category_list"] == cat]
        st.write(
            f"{cat} ({len(subset)} postings, Avg Salary ${subset['average_salary'].mean():,.0f})"
        )
        st.dataframe(subset["key_role"].value_counts().head(10).to_frame("Count"))

with tab3:
    st.subheader("Time Series Trends")

    monthly = filtered["year_month"].value_counts().sort_index().reset_index()
    monthly.columns = ["Year-Month", "Postings"]
    fig_monthly = px.line(monthly, x="Year-Month", y="Postings", markers=True, title="Monthly Posting Volume")
    st.plotly_chart(fig_monthly, use_container_width=True)

    df_exploded = filtered.explode("category_list")
    df_exploded = df_exploded[df_exploded["category_list"].notna()]
    df_exploded = df_exploded[df_exploded["category_list"].str.strip() != ""]

    top_cats = df_exploded["category_list"].value_counts().head(8).index
    cat_monthly = (
        df_exploded[df_exploded["category_list"].isin(top_cats)]
        .groupby(["year_month", "category_list"])
        .size()
        .reset_index(name="postings")
    )

    fig_cat_trend = px.line(
        cat_monthly,
        x="year_month",
        y="postings",
        color="category_list",
        markers=True,
        title="Monthly Trends - Top Categories",
    )
    st.plotly_chart(fig_cat_trend, use_container_width=True)

    top_roles = filtered["key_role"].value_counts().head(10).index
    role_monthly = (
        filtered[filtered["key_role"].isin(top_roles)]
        .groupby(["year_month", "key_role"])
        .size()
        .reset_index(name="postings")
    )

    fig_role_trend = px.line(
        role_monthly,
        x="year_month",
        y="postings",
        color="key_role",
        markers=True,
        title="Monthly Trends - Top Key Roles",
    )
    st.plotly_chart(fig_role_trend, use_container_width=True)

with tab4:
    st.subheader("Quarter-over-Quarter Growth")

    col_min_cat, col_min_role, col_top_n = st.columns(3)
    with col_min_cat:
        min_volume_cat = st.number_input(
            "Min postings per category (QoQ)",
            min_value=0,
            max_value=10000,
            value=100,
            step=10,
        )
    with col_min_role:
        min_volume_role = st.number_input(
            "Min postings per role (QoQ)",
            min_value=0,
            max_value=10000,
            value=100,
            step=10,
        )
    with col_top_n:
        top_n = st.slider("Top N rows", min_value=5, max_value=50, value=10, step=5)

    quarters_sorted = sorted(filtered["quarter"].unique())
    if len(quarters_sorted) < 2:
        st.info("Not enough quarters in the current filters to compute QoQ growth.")
    else:
        transitions = list(zip(quarters_sorted[:-1], quarters_sorted[1:]))

        def qoq_table(exploded_df, group_col, min_volume=100):
            quarterly_counts = exploded_df.groupby(["quarter", group_col]).size().reset_index(name="count")
            result = {}
            for prev_q, next_q in transitions:
                prev = quarterly_counts[quarterly_counts["quarter"] == prev_q].set_index(group_col)["count"]
                next_ = quarterly_counts[quarterly_counts["quarter"] == next_q].set_index(group_col)["count"]
                all_groups = prev.index.union(next_.index)
                prev = prev.reindex(all_groups, fill_value=0)
                next_ = next_.reindex(all_groups, fill_value=0)

                qoq = pd.DataFrame({"Prev": prev, "Next": next_})
                qoq = qoq[(qoq["Prev"] >= min_volume) | (qoq["Next"] >= min_volume)]
                qoq["Growth %"] = ((qoq["Next"] - qoq["Prev"]) / qoq["Prev"].replace(0, 1)) * 100
                qoq = qoq.sort_values("Growth %", ascending=False)
                result[f"{prev_q} -> {next_q}"] = qoq
            return result

        df_exploded_cat = filtered.explode("category_list")
        df_exploded_cat = df_exploded_cat[df_exploded_cat["category_list"].notna()]
        df_exploded_cat = df_exploded_cat[df_exploded_cat["category_list"].str.strip() != ""]

        df_exploded_role = filtered[filtered["key_role"].notna()].copy()

        cat_qoq = qoq_table(df_exploded_cat, "category_list", min_volume=min_volume_cat)
        role_qoq = qoq_table(df_exploded_role, "key_role", min_volume=min_volume_role)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for label, table in cat_qoq.items():
                zf.writestr(
                    f"qoq_categories_{label.replace(' ', '').replace('->', '_')}.csv",
                    table.to_csv(index=True),
                )
            for label, table in role_qoq.items():
                zf.writestr(
                    f"qoq_roles_{label.replace(' ', '').replace('->', '_')}.csv",
                    table.to_csv(index=True),
                )

        st.download_button(
            label="Download all QoQ tables (zip)",
            data=zip_buffer.getvalue(),
            file_name="qoq_tables.zip",
            mime="application/zip",
        )

        st.markdown("**Category QoQ Growth**")
        for label, table in cat_qoq.items():
            st.markdown(f"{label}")
            csv_data = table.to_csv(index=True)
            st.download_button(
                label=f"Download categories QoQ ({label})",
                data=csv_data,
                file_name=f"qoq_categories_{label.replace(' ', '').replace('->', '_')}.csv",
                mime="text/csv",
                key=f"qoq_cat_{label}",
            )
            st.dataframe(table.head(top_n))
            st.dataframe(table.tail(top_n))

        st.markdown("**Key Role QoQ Growth**")
        for label, table in role_qoq.items():
            st.markdown(f"{label}")
            csv_data = table.to_csv(index=True)
            st.download_button(
                label=f"Download roles QoQ ({label})",
                data=csv_data,
                file_name=f"qoq_roles_{label.replace(' ', '').replace('->', '_')}.csv",
                mime="text/csv",
                key=f"qoq_role_{label}",
            )
            st.dataframe(table.head(top_n))
            st.dataframe(table.tail(top_n))

with tab5:
    st.subheader("Compensation Intelligence")

    view_mode = st.radio("View", ["Charts", "Tables"], horizontal=True)
    chart_type = st.selectbox("Chart type", ["Box", "Violin", "Histogram"], index=0)
    show_all_charts = st.checkbox("Show all charts", value=False)

    st.markdown("**Top 10 Highest Paying Roles**")
    top_paying = filtered.nlargest(10, "average_salary")[
        ["title", "average_salary", "positionLevels", "minimumYearsExperience"]
    ]

    salary_means = (
        filtered.groupby("positionLevels")["average_salary"].mean().sort_values(ascending=False).reset_index()
    )
    category_salary = (
        filtered.groupby("primary_category")["average_salary"].mean().sort_values(ascending=False).reset_index()
    )

    if view_mode == "Charts":
        st.dataframe(top_paying)

        st.markdown("**Salary Distribution by Position Level**")
        if show_all_charts or chart_type == "Box":
            fig_dist = px.box(
                filtered,
                x="positionLevels",
                y="average_salary",
                color="positionLevels",
                title="Salary Distribution by Position Level (Box)",
            )
            fig_dist.update_layout(showlegend=False, xaxis={"tickangle": 45})
            st.plotly_chart(fig_dist, use_container_width=True)
        if show_all_charts or chart_type == "Violin":
            fig_dist = px.violin(
                filtered,
                x="positionLevels",
                y="average_salary",
                color="positionLevels",
                box=True,
                points="outliers",
                title="Salary Distribution by Position Level (Violin)",
            )
            fig_dist.update_layout(showlegend=False, xaxis={"tickangle": 45})
            st.plotly_chart(fig_dist, use_container_width=True)
        if show_all_charts or chart_type == "Histogram":
            fig_dist = px.histogram(
                filtered,
                x="average_salary",
                nbins=30,
                color="positionLevels",
                title="Salary Distribution (Histogram)",
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        fig_bar = px.bar(
            salary_means,
            x="average_salary",
            y="positionLevels",
            orientation="h",
            title="Average Salary by Position Level",
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_cat = px.bar(
            category_salary.head(15),
            x="average_salary",
            y="primary_category",
            orientation="h",
            title="Average Salary by Category (Top 15)",
        )
        fig_cat.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.dataframe(top_paying)
        st.markdown("**Average Salary by Position Level**")
        st.dataframe(salary_means)
        st.markdown("**Average Salary by Category**")
        st.dataframe(category_salary)

with tab6:
    st.subheader("Headhunter Opportunity Matrix")

    title_stats = filtered.groupby("clean_title").agg(
        postings=("metadata_jobPostId", "count"),
        avg_salary=("average_salary", "mean"),
        avg_reposts=("metadata_repostCount", "mean"),
        avg_apps=("metadata_totalNumberJobApplication", "mean"),
    )
    title_stats = title_stats[title_stats["postings"] > 5].copy()
    title_stats["opportunity_score"] = (
        title_stats["avg_salary"] * (title_stats["avg_reposts"] + 1)
    ) / (title_stats["avg_apps"] + 1)

    top_opps = title_stats.sort_values("opportunity_score", ascending=False).head(20)
    st.dataframe(
        top_opps[["postings", "avg_salary", "avg_reposts", "avg_apps", "opportunity_score"]]
        .round(1)
    )

    fig_scatter = px.scatter(
        title_stats,
        x="avg_apps",
        y="avg_salary",
        size="postings",
        color="avg_reposts",
        hover_name=title_stats.index,
        title="Salary vs Competition (color = reposts)",
        labels={"avg_apps": "Avg Applications", "avg_salary": "Avg Salary"},
        color_continuous_scale="Reds",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab7:
    st.subheader("Quick-to-Fill vs Hard-to-Fill Roles")

    quick_fill = filtered[
        (filtered["metadata_totalNumberJobApplication"] > 10) & (filtered["metadata_repostCount"] == 0)
    ].sort_values("metadata_totalNumberJobApplication", ascending=False)

    hard_fill = filtered[
        (filtered["metadata_repostCount"] > 1)
        | ((filtered["metadata_totalNumberJobApplication"] == 0) & (filtered["metadata_totalNumberOfView"] > 50))
    ].sort_values("metadata_repostCount", ascending=False)

    col_q, col_h = st.columns(2)
    with col_q:
        st.markdown("**Quick-to-Fill Roles**")
        st.dataframe(
            quick_fill[["title", "metadata_totalNumberJobApplication", "average_salary", "employmentTypes"]].head(15)
        )
    with col_h:
        st.markdown("**Hard-to-Fill Roles**")
        st.dataframe(
            hard_fill[["title", "metadata_repostCount", "metadata_totalNumberJobApplication", "average_salary"]].head(15)
        )

    q_cats = Counter([cat for cats in quick_fill["category_list"] for cat in cats]).most_common(10)
    h_cats = Counter([cat for cats in hard_fill["category_list"] for cat in cats]).most_common(10)

    col_qc, col_hc = st.columns(2)
    with col_qc:
        st.markdown("**Quick-to-Fill Categories**")
        st.dataframe(pd.DataFrame(q_cats, columns=["Category", "Count"]))
    with col_hc:
        st.markdown("**Hard-to-Fill Categories**")
        st.dataframe(pd.DataFrame(h_cats, columns=["Category", "Count"]))

with tab8:
    st.subheader("Job Seeker Explorer")
    search_term = st.text_input("Search a role (e.g., analyst, manager)")

    if search_term:
        results = filtered[filtered["clean_title"].str.contains(search_term.lower())]
        if results.empty:
            st.warning("No results found. Try a broader term.")
        else:
            st.write(f"Found {len(results)} postings")
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Salary IQR",
                f"${int(results['salary_minimum'].quantile(0.25)):,} - ${int(results['salary_maximum'].quantile(0.75)):,}",
            )
            c2.metric("Top Industry", results["primary_category"].mode()[0])
            c3.metric("Avg Applications", f"{int(results['metadata_totalNumberJobApplication'].mean())}")

            st.dataframe(
                results[["title", "postedCompany_name", "average_salary", "employmentTypes"]].head(15)
            )
    else:
        all_titles = " ".join(filtered["clean_title"].dropna().tolist())
        words = all_titles.split()
        bigrams = zip(words, words[1:])
        bigram_counts = Counter(bigrams)
        top_bigrams = bigram_counts.most_common(12)
        bigram_df = pd.DataFrame(top_bigrams, columns=["Phrase", "Count"])
        bigram_df["Phrase"] = bigram_df["Phrase"].apply(lambda x: f"{x[0]} {x[1]}")

        fig_trend = px.bar(bigram_df, x="Count", y="Phrase", orientation="h", title="Top Role Phrases")
        fig_trend.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")
st.caption("NTU DSAI Module 1 Project | Data Source: SGJobData.csv")