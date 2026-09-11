# Customer-Churn-Analysis
# ============================================================
# CUSTOMER CHURN ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_name = "customer_churn.csv"

df = pd.read_csv(file_name)

print("Dataset loaded successfully!")
print("=" * 60)


# ============================================================
# 2. BASIC DATA INSPECTION
# ============================================================

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nCleaned column names:")
print(df.columns.tolist())


# ============================================================
# 4. CLEAN COMMON TELCO DATASET
# ============================================================

# Convert TotalCharges to numeric if available

if "totalcharges" in df.columns:

    df["totalcharges"] = pd.to_numeric(
        df["totalcharges"],
        errors="coerce"
    )


# Convert SeniorCitizen to categorical text

if "seniorcitizen" in df.columns:

    df["seniorcitizen"] = df["seniorcitizen"].map({
        0: "No",
        1: "Yes"
    })


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    df[column] = df[column].fillna("Unknown")


print("\nMissing values handled.")


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("\nDuplicates removed:", before - after)


# ============================================================
# 7. CHECK CHURN COLUMN
# ============================================================

if "churn" not in df.columns:

    raise ValueError(
        "Churn column not found. "
        "Make sure your dataset contains a column named 'Churn'."
    )


# Convert Churn into 0 and 1

df["churn_flag"] = df["churn"].map({
    "Yes": 1,
    "No": 0
})


# Handle already numeric churn values

df["churn_flag"] = df["churn_flag"].fillna(
    pd.to_numeric(df["churn"], errors="coerce")
)

df["churn_flag"] = df["churn_flag"].fillna(0)


# ============================================================
# 8. BASELINE CHURN RATE
# ============================================================

total_customers = len(df)

churned_customers = df["churn_flag"].sum()

churn_rate = (
    churned_customers /
    total_customers
) * 100


print("\n")
print("=" * 60)
print("BASELINE CHURN ANALYSIS")
print("=" * 60)

print("Total Customers :", total_customers)
print("Churned Customers:", int(churned_customers))
print("Churn Rate       :", round(churn_rate, 2), "%")


# ============================================================
# 9. MONTHLY RECURRING REVENUE (MRR)
# ============================================================

if "monthlycharges" in df.columns:

    total_mrr = df["monthlycharges"].sum()

    churned_mrr = df.loc[
        df["churn_flag"] == 1,
        "monthlycharges"
    ].sum()

    mrr_loss_percentage = (
        churned_mrr /
        total_mrr
    ) * 100

else:

    total_mrr = 0
    churned_mrr = 0
    mrr_loss_percentage = 0


print("\nMRR ANALYSIS")
print("-" * 60)

print(
    "Total Monthly Revenue : ₹",
    round(total_mrr, 2)
)

print(
    "Monthly Revenue Lost  : ₹",
    round(churned_mrr, 2)
)

print(
    "Percentage MRR Lost   :",
    round(mrr_loss_percentage, 2),
    "%"
)


# ============================================================
# 10. CUSTOMER LIFETIME VALUE (CLV)
# ============================================================

if "monthlycharges" in df.columns:

    if "tenure" in df.columns:

        # Simple CLV approximation
        df["clv"] = (
            df["monthlycharges"] *
            df["tenure"]
        )

    else:

        # Assume average lifetime of 12 months
        df["clv"] = (
            df["monthlycharges"] *
            12
        )

else:

    df["clv"] = 0


average_clv = df["clv"].mean()

churned_clv = df.loc[
    df["churn_flag"] == 1,
    "clv"
].sum()


print("\nCLV ANALYSIS")
print("-" * 60)

print(
    "Average Customer Lifetime Value : ₹",
    round(average_clv, 2)
)

print(
    "Total CLV of Churned Customers  : ₹",
    round(churned_clv, 2)
)


# ============================================================
# 11. CHURN BY CONTRACT TYPE
# ============================================================

if "contract" in df.columns:

    contract_churn = (
        df.groupby("contract")["churn_flag"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print("\nCHURN RATE BY CONTRACT")
    print(contract_churn.round(2))


# ============================================================
# 12. CHURN BY TENURE
# ============================================================

if "tenure" in df.columns:

    def tenure_group(months):

        if months <= 12:
            return "0-12 Months"

        elif months <= 24:
            return "13-24 Months"

        elif months <= 48:
            return "25-48 Months"

        else:
            return "49+ Months"


    df["tenure_group"] = df["tenure"].apply(
        tenure_group
    )


    tenure_churn = (
        df.groupby("tenure_group")["churn_flag"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print("\nCHURN RATE BY TENURE")
    print(tenure_churn.round(2))


# ============================================================
# 13. CHURN BY PAYMENT METHOD
# ============================================================

if "paymentmethod" in df.columns:

    payment_churn = (
        df.groupby("paymentmethod")["churn_flag"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print("\nCHURN RATE BY PAYMENT METHOD")
    print(payment_churn.round(2))


# ============================================================
# 14. CHURN BY INTERNET SERVICE
# ============================================================

if "internetservice" in df.columns:

    internet_churn = (
        df.groupby("internetservice")["churn_flag"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print("\nCHURN RATE BY INTERNET SERVICE")
    print(internet_churn.round(2))


# ============================================================
# 15. CHURN BY SENIOR CITIZEN
# ============================================================

if "seniorcitizen" in df.columns:

    senior_churn = (
        df.groupby("seniorcitizen")["churn_flag"]
        .mean()
        .mul(100)
    )

    print("\nCHURN RATE BY SENIOR CITIZEN")
    print(senior_churn.round(2))


# ============================================================
# 16. VISUALIZATION 1
#     OVERALL CHURN
# ============================================================

plt.figure(figsize=(7, 5))

churn_counts = df["churn"].value_counts()

plt.bar(
    churn_counts.index,
    churn_counts.values
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "01_overall_churn.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. VISUALIZATION 2
#     CHURN BY CONTRACT
# ============================================================

if "contract" in df.columns:

    plt.figure(figsize=(8, 5))

    contract_churn.plot(
        kind="bar"
    )

    plt.title(
        "Churn Rate by Contract Type"
    )

    plt.xlabel("Contract Type")

    plt.ylabel(
        "Churn Rate (%)"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        "02_churn_by_contract.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 18. VISUALIZATION 3
#     CHURN BY TENURE
# ============================================================

if "tenure" in df.columns:

    plt.figure(figsize=(8, 5))

    tenure_churn.plot(
        kind="bar"
    )

    plt.title(
        "Churn Rate by Tenure"
    )

    plt.xlabel(
        "Tenure Group"
    )

    plt.ylabel(
        "Churn Rate (%)"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        "03_churn_by_tenure.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 19. VISUALIZATION 4
#     CHURN BY PAYMENT METHOD
# ============================================================

if "paymentmethod" in df.columns:

    plt.figure(figsize=(10, 5))

    payment_churn.plot(
        kind="bar"
    )

    plt.title(
        "Churn Rate by Payment Method"
    )

    plt.xlabel(
        "Payment Method"
    )

    plt.ylabel(
        "Churn Rate (%)"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        "04_churn_by_payment.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 20. VISUALIZATION 5
#     CLV BOXPLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="churn",
    y="clv",
    data=df
)

plt.title(
    "Customer Lifetime Value by Churn Status"
)

plt.xlabel(
    "Churn"
)

plt.ylabel(
    "Customer Lifetime Value"
)

plt.tight_layout()

plt.savefig(
    "05_clv_boxplot.png",
    dpi=300
)

plt.show()


# ============================================================
# 21. VISUALIZATION 6
#     CORRELATION HEATMAP
# ============================================================

numeric_data = df.select_dtypes(
    include=["int64", "float64"]
)

correlation = numeric_data.corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    "06_correlation_heatmap.png",
    dpi=300
)

plt.show()


# ============================================================
# 22. IDENTIFY HIGH-CHURN CONTRACT
# ============================================================

highest_contract = "Not available"

if "contract" in df.columns:

    highest_contract = (
        contract_churn.idxmax()
    )

    highest_contract_rate = (
        contract_churn.max()
    )

    print("\nHighest churn contract:")
    print(
        highest_contract,
        "-",
        round(highest_contract_rate, 2),
        "%"
    )


# ============================================================
# 23. IDENTIFY HIGH-CHURN PAYMENT METHOD
# ============================================================

highest_payment = "Not available"

if "paymentmethod" in df.columns:

    highest_payment = (
        payment_churn.idxmax()
    )

    highest_payment_rate = (
        payment_churn.max()
    )

    print("\nHighest churn payment method:")
    print(
        highest_payment,
        "-",
        round(highest_payment_rate, 2),
        "%"
    )


# ============================================================
# 24. IDENTIFY HIGH-RISK TENURE GROUP
# ============================================================

highest_tenure = "Not available"

if "tenure" in df.columns:

    highest_tenure = (
        tenure_churn.idxmax()
    )

    highest_tenure_rate = (
        tenure_churn.max()
    )

    print("\nHighest churn tenure group:")
    print(
        highest_tenure,
        "-",
        round(highest_tenure_rate, 2),
        "%"
    )


# ============================================================
# 25. BUSINESS RECOMMENDATIONS
# ============================================================

print("\n")
print("=" * 60)
print("BUSINESS RECOMMENDATIONS")
print("=" * 60)

print("""
1. Target customers with high churn probability
   using personalized retention offers.

2. Encourage customers to move from short-term
   contracts to long-term contracts.

3. Provide special offers during the first year
   to improve customer retention.

4. Investigate payment methods with higher churn
   and make payment processes easier.

5. Improve customer service and technical support
   for customers showing signs of dissatisfaction.

6. Offer loyalty discounts to customers with
   high Customer Lifetime Value (CLV).

7. Monitor monthly recurring revenue from customers
   at high risk of churn.

8. Use regular customer feedback surveys to identify
   problems before customers cancel their subscriptions.
""")


# ============================================================
# 26. CREATE SUMMARY TABLE
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Total Customers",
        "Churned Customers",
        "Churn Rate (%)",
        "Total MRR",
        "MRR Loss",
        "MRR Loss (%)",
        "Average CLV",
        "Churned Customer CLV"
    ],

    "Value": [
        total_customers,
        int(churned_customers),
        round(churn_rate, 2),
        round(total_mrr, 2),
        round(churned_mrr, 2),
        round(mrr_loss_percentage, 2),
        round(average_clv, 2),
        round(churned_clv, 2)
    ]
})


print("\nSUMMARY TABLE")
print(summary)


# ============================================================
# 27. EXPORT SUMMARY
# ============================================================

summary.to_csv(
    "churn_analysis_summary.csv",
    index=False
)

print(
    "\nSummary exported as "
    "churn_analysis_summary.csv"
)


# ============================================================
# 28. EXPORT CLEANED DATASET
# ============================================================

df.to_csv(
    "customer_churn_cleaned.csv",
    index=False
)

print(
    "Cleaned dataset exported as "
    "customer_churn_cleaned.csv"
)


# ============================================================
# 29. CREATE PDF ANALYTICS REPORT
# ============================================================

pdf_file = "Customer_Churn_Analytics_Report.pdf"

doc = SimpleDocTemplate(
    pdf_file,
    pagesize=A4,
    rightMargin=40,
    leftMargin=40,
    topMargin=40,
    bottomMargin=40
)

styles = getSampleStyleSheet()

story = []


# TITLE

story.append(
    Paragraph(
        "Customer Churn Analysis Report",
        styles["Title"]
    )
)

story.append(
    Spacer(1, 15)
)


# INTRODUCTION

story.append(
    Paragraph(
        "This report analyzes customer churn patterns "
        "and identifies important factors associated "
        "with subscription cancellations.",
        styles["BodyText"]
    )
)

story.append(
    Spacer(1, 15)
)


# SUMMARY

story.append(
    Paragraph(
        "Key Metrics",
        styles["Heading2"]
    )
)

table_data = [
    ["Metric", "Value"]
]

for _, row in summary.iterrows():

    table_data.append([
        str(row["Metric"]),
        str(row["Value"])
    ])


table = Table(
    table_data,
    colWidths=[250, 150]
)

table.setStyle(
    TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),

        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.black
        ),

        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey
        ),

        (
            "ALIGN",
            (1, 1),
            (-1, -1),
            "CENTER"
        ),

        (
            "PADDING",
            (0, 0),
            (-1, -1),
            6
        )
    ])
)

story.append(table)

story.append(
    Spacer(1, 20)
)


# FINDINGS

story.append(
    Paragraph(
        "Key Findings",
        styles["Heading2"]
    )
)

findings = f"""
• Overall customer churn rate is
{round(churn_rate, 2)}%.

• Total monthly recurring revenue is
₹{round(total_mrr, 2)}.

• Monthly recurring revenue lost because
of churn is ₹{round(churned_mrr, 2)}.

• Average Customer Lifetime Value is
₹{round(average_clv, 2)}.

• The contract type with the highest churn
is {highest_contract}.

• The payment method with the highest churn
is {highest_payment}.

• The highest-risk tenure group is
{highest_tenure}.
"""

story.append(
    Paragraph(
        findings.replace("\n", "<br/>"),
        styles["BodyText"]
    )
)

story.append(
    Spacer(1, 20)
)


# CHARTS

chart_files = [
    "01_overall_churn.png",
    "02_churn_by_contract.png",
    "03_churn_by_tenure.png",
    "04_churn_by_payment.png",
    "05_clv_boxplot.png",
    "06_correlation_heatmap.png"
]


for chart in chart_files:

    if os.path.exists(chart):

        story.append(
            Image(
                chart,
                width=6.5 * inch,
                height=4.0 * inch
            )
        )

        story.append(
            Spacer(1, 15)
        )


# RECOMMENDATIONS

story.append(
    Paragraph(
        "Business Recommendations",
        styles["Heading2"]
    )
)

recommendations = [
    "Create personalized retention offers.",
    "Encourage customers to choose long-term contracts.",
    "Provide special offers to new customers.",
    "Improve payment and billing experience.",
    "Improve customer service and technical support.",
    "Provide loyalty benefits to high-value customers.",
    "Monitor customers who show signs of churn.",
    "Collect customer feedback regularly."
]

for recommendation in recommendations:

    story.append(
        Paragraph(
            "• " + recommendation,
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 5)
    )


# BUILD PDF

doc.build(story)

print("\n")
print("=" * 60)
print("PDF REPORT CREATED SUCCESSFULLY!")
print("File:", pdf_file)
print("=" * 60)


# ============================================================
# 30. FINAL MESSAGE
# ============================================================

print("""
PROJECT COMPLETED SUCCESSFULLY!

Generated files:

1. customer_churn_cleaned.csv
2. churn_analysis_summary.csv
3. Customer_Churn_Analytics_Report.pdf

Charts:

1. 01_overall_churn.png
2. 02_churn_by_contract.png
3. 03_churn_by_tenure.png
4. 04_churn_by_payment.png
5. 05_clv_boxplot.png
6. 06_correlation_heatmap.png
""")