import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORT_DIR = os.path.join(BASE_DIR, 'report')
IMAGES_DIR = os.path.join(REPORT_DIR, 'images')
STATS_DIR = os.path.join(REPORT_DIR, 'stats')
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'docs'))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'Interim_Fraud_Detection_Report.pdf')

def create_report():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Using smaller margins to fit more content
    doc = SimpleDocTemplate(OUTPUT_FILE, pagesize=letter, topMargin=0.4*inch, bottomMargin=0.4*inch, leftMargin=0.5*inch, rightMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    body_style = styles['Normal']
    caption_style = ParagraphStyle('Caption', parent=styles['Normal'], alignment=1, italic=True, fontSize=8, spaceBefore=4, spaceAfter=8)
    tech_style = ParagraphStyle('Technical', parent=styles['Normal'], fontSize=9, leading=11, spaceAfter=6)
    bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'], leftIndent=20, firstLineIndent=0, spaceBefore=3)

    # 1. Header
    story.append(Paragraph("Interim Technical Report: Advanced Fraud Detection System", title_style))
    story.append(Paragraph("<b>Client:</b> Adey Innovations Inc.", styles['Heading3']))
    story.append(Paragraph("<b>Author:</b> Biruk Gebru Jember", body_style))
    story.append(Paragraph("<b>Date:</b> December 2025", body_style))
    story.append(Spacer(1, 0.15*inch))

    # 2. Business Objective & Impact
    story.append(Paragraph("1. Business Objective & Strategic Impact", heading_style))
    story.append(Paragraph(
        "For Adey Innovations Inc., the deployment of a high-precision fraud detection system is not merely a technical requirement but a strategic necessity. "
        "Implementing accurate fraud detection <b>directly prevents substantial financial losses</b> by identifying and blocking fraudulent transactions before they are processed. "
        "More importantly, it <b>builds and maintains customer trust</b>; by ensuring that legitimate users are rarely inconvenienced by false flags while their accounts remain protected from intruders. "
        "The project addresses two distinct domains: high-volume E-commerce transactions and sensitive Bank credit card data, both characterized by extreme class imbalance and evolving fraud patterns.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    # 3. Task 1: Comprehensive Data Analysis & Preprocessing
    story.append(Paragraph("2. Task 1: Comprehensive Data Analysis & Preprocessing", heading_style))
    story.append(Paragraph(
        "<b>2.1 Exploratory Data Analysis (EDA) & Bivariate Insights</b>", subheading_style
    ))
    story.append(Paragraph(
        "Our EDA revealed critical differences between legitimate and fraudulent behaviors. <b>Figure 1</b> illustrates the age distribution: "
        "fraudulent actors are often distributed differently across age groups compared to standard consumers. Bivariate analysis "
        "between 'Purchase Value' and 'Class' (<b>Figure 2</b>) shows that fraudulent transactions often involve outlier amounts designed to max out accounts quickly. "
        "Furthermore, our mapping of IP addresses to countries (<b>Figure 3</b>) highlighted specific geographic clusters with disproportionately high fraud rates.",
        tech_style
    ))

    # Visual Cluster 1 (EDA)
    img_width = 3.2*inch
    img_height = 2*inch
    
    # Age & Purchase Value Row
    data = []
    row = []
    if os.path.exists(os.path.join(IMAGES_DIR, 'fraud_age_distribution.png')):
        row.append(Image(os.path.join(IMAGES_DIR, 'fraud_age_distribution.png'), width=img_width, height=img_height))
    if os.path.exists(os.path.join(IMAGES_DIR, 'fraud_purchase_value_box.png')):
        row.append(Image(os.path.join(IMAGES_DIR, 'fraud_purchase_value_box.png'), width=img_width, height=img_height))
    
    if row:
        t = Table([row], colWidths=[img_width+0.2*inch]*2)
        story.append(t)
        story.append(Paragraph("Figure 1 (Left): Age Distribution Disparities | Figure 2 (Right): Purchase Value Outlier Detection", caption_style))

    story.append(Paragraph(
        "<b>2.2 Strategic Feature Engineering</b>", subheading_style
    ))
    story.append(Paragraph(
        "To capture the temporal and behavioral signatures of fraud, we engineered several key features:",
        tech_style
    ))
    story.append(Paragraph("• <b>Hour of Day & Day of Week:</b> Extracted to detect automated 'bot' activity typical during low-traffic overnight hours.", bullet_style))
    story.append(Paragraph("• <b>Time Since Signup:</b> Measures the velocity between registration and transaction. We found a high density of fraud in 'instant-purchase' profiles.", bullet_style))
    story.append(Paragraph("• <b>Transaction Frequency (Velocity):</b> Calculated per Device ID and IP address to identify rapid-fire attacks from a single source.", bullet_style))
    
    story.append(Paragraph(
        "<b>2.3 Data Transformation Pipeline</b>", subheading_style
    ))
    story.append(Paragraph(
        "To prepare features for ensemble learning, we implemented a robust pipeline: <b>StandardScaler</b> was applied to all numeric features to normalize "
        "scale differences (e.g., Age vs. Purchase Value), while <b>One-Hot Encoding</b> converted categorical variables like Browser and Gender into model-ready vectors. "
        "Correlation analysis (<b>Figure 4</b>) ensured feature independence to prevent multicollinearity in our baseline models.",
        tech_style
    ))

    # Correlation & Country Map Row
    row = []
    if os.path.exists(os.path.join(IMAGES_DIR, 'creditcard_correlation.png')):
        row.append(Image(os.path.join(IMAGES_DIR, 'creditcard_correlation.png'), width=img_width, height=img_height))
    if os.path.exists(os.path.join(IMAGES_DIR, 'top_fraud_countries.png')):
        row.append(Image(os.path.join(IMAGES_DIR, 'top_fraud_countries.png'), width=img_width, height=img_height))
    
    if row:
        t = Table([row], colWidths=[img_width+0.2*inch]*2)
        story.append(t)
        story.append(Paragraph("Figure 4 (Left): Feature Correlation Matrix | Figure 3 (Right): Geographic Fraud Hotspots", caption_style))

    # 4. Class Imbalance & Evaluation Metrics
    story.append(Paragraph("3. Managing Class Imbalance & Model Stability", heading_style))
    story.append(Paragraph(
        "As detailed in <b>Table 1</b> and <b>Figure 5</b>, both datasets suffer from severe class imbalance. For the banking dataset, fraudulent "
        "cases represent less than 0.2% of the data. We addressed this using <b>SMOTE (Synthetic Minority Over-sampling Technique)</b> during training, "
        "allowing the model to learn fraud boundaries without simple repetition bias.",
        tech_style
    ))

    # Table 1: Raw Distribution Summary
    try:
        fraud_raw = pd.read_csv(os.path.join(STATS_DIR, 'fraud_data_raw_stats.csv'), index_col=0)
        mean_val = fraud_raw.loc['mean', 'class']
        data = [
            ["Dataset", "Total Samples", "Fraud %", "Methodology"],
            ["E-commerce Fraud", "151,112", f"{mean_val*100:.2f}%", "SMOTE (Balanced)"],
            ["Bank Transactions", "284,807", "0.17%", "SMOTE (Balanced)"]
        ]
        t = Table(data, hAlign='CENTER')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
        ]))
        story.append(t)
        story.append(Paragraph("Table 1: Formally Referenced Dataset Composition and Imbalance Strategy", caption_style))
    except:
        pass

    # Class Distributions (Fraud & Credit)
    row = []
    if os.path.exists(os.path.join(IMAGES_DIR, 'fraud_class_distribution.png')):
        row.append(Image(os.path.join(IMAGES_DIR, 'fraud_class_distribution.png'), width=2.5*inch, height=1.5*inch))
    if os.path.exists(os.path.join(IMAGES_DIR, 'creditcard_class_distribution.png')):
        row.append(Image(os.path.join(IMAGES_DIR, 'creditcard_class_distribution.png'), width=2.5*inch, height=1.5*inch))
    
    if row:
        t = Table([row], colWidths=[2.7*inch]*2)
        story.append(t)
        story.append(Paragraph("Figure 5: Visualizing Class Imbalance across E-commerce and Bank Data", caption_style))

    # 5. Task 2 Results: Model Performance
    story.append(Paragraph("4. Task 2: Comparative Model Performance", heading_style))
    story.append(Paragraph(
        "We executed a rigorous training cycle using <b>Stratified 5-Fold Cross-Validation</b>. As summarized in <b>Table 2</b> "
        "and visualized in <b>Figure 6</b>, XGBoost provides the most stable separation, outperforming the linear baseline. "
        "The XGBoost model achieved an <b>ROC AUC of ~0.84</b>, effectively capturing the complex, non-linear dependencies between behavioral features.",
        tech_style
    ))

    # Results Table
    try:
        results_df = pd.read_csv(os.path.join(STATS_DIR, 'model_comparison_results.csv'), index_col=0)
        data = [["Model Name", "ROC AUC Score"]] + results_df.values.tolist()
        t = Table(data, hAlign='CENTER')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
        ]))
        story.append(t)
        story.append(Paragraph("Table 2: Side-by-Side Model Metric Comparison", caption_style))
    except:
        pass

    if os.path.exists(os.path.join(IMAGES_DIR, 'roc_curve_comparison.png')):
        story.append(Image(os.path.join(IMAGES_DIR, 'roc_curve_comparison.png'), width=4.5*inch, height=2.5*inch))
        story.append(Paragraph("Figure 6: ROC Curve Performance Analysis per Model Architecture", caption_style))

    # 6. Future Roadmap
    story.append(Paragraph("5. Future Roadmap & Risk Mitigation", heading_style))
    story.append(Paragraph(
        "With a high-performing model established, our next phases will transition from prediction to interpretation and business action.",
        body_style
    ))
    
    story.append(Paragraph("<b>5.1 Task 3: Interpretability (SHAP)</b>", subheading_style))
    story.append(Paragraph(
        "We will implement <b>SHAP (SHapley Additive exPlanations)</b> to decompose predictions. Our roadmap includes:",
        tech_style
    ))
    story.append(Paragraph("• <b>Global Feature Importance:</b> Identifying the top 5 global drivers of fraud for policy-level decisions.", bullet_style))
    story.append(Paragraph("• <b>Individual Prediction Explanations:</b> Visualizing 'Force Plots' for specific flagged accounts to assist manual review teams.", bullet_style))
    story.append(Paragraph("• <b>Challenge & Mitigation:</b> SHAP computation is resource-intensive for large datasets. We will mitigate this using specialized TreeSHAP kernels and background dataset sampling.", bullet_style))

    story.append(Paragraph("<b>5.2 Business Recommendations & Final Validation</b>", subheading_style))
    story.append(Paragraph(
        "Planned deliverables include precise rule-based thresholds derived from SHAP insights (e.g., transaction-delay policies). "
        "Finally, we will validate the cross-platform applicability of the model by testing its transferability between "
        "e-commerce behavior and banking transaction signatures.",
        tech_style
    ))

    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Report concludes the interim milestone for the Adey Innovations Inc. Fraud Detection Project.", body_style))

    # Build PDF
    doc.build(story)
    print(f"Final Revised Report successfully generated at: {OUTPUT_FILE}")

if __name__ == '__main__':
    create_report()
