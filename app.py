from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['STATIC_FOLDER'] = "static"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

pipeline = joblib.load("exoplanet_pipeline_3class (3).pkl")
feature_cols = joblib.load("kepler_features (1).pkl")
X_train = pd.read_csv("X_train_sample.csv")

label_map = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}
reverse_label_map = {v: k for k, v in label_map.items()}


DARK_BG_HEX = "#101022"  
plt.style.use('dark_background')

plt.rcParams.update({
    "figure.facecolor": DARK_BG_HEX,
    "axes.facecolor": DARK_BG_HEX,
    "savefig.facecolor": DARK_BG_HEX,
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "legend.facecolor": "#1e1e3f",
    "legend.edgecolor": "white",
})

CLASS_COLORS = ['#7c3aed', '#06b6d4', '#f97316']  


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results', methods=['POST'])
def results():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    # Save uploaded CSV
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    df = pd.read_csv(file_path)


    X = df.copy()
    for col in feature_cols:
        if col not in X.columns:
          
            if col in X_train.columns:
                X[col] = X_train[col].median()
            else:
                X[col] = 0
    X = X[feature_cols]
    X = X.fillna(X_train.median())

    # --- Make predictions ---
    preds = pipeline.predict(X)
    df['Predicted Disposition'] = [label_map[p] for p in preds]

    # --- EDA Charts ---
    # Ensure seaborn uses a dark-friendly style but keep control via matplotlib rcParams
    sns.set_style("darkgrid", {"axes.facecolor": DARK_BG_HEX, "grid.color": "#22223a"})

    chart_files = {}

    # ---------- Bar chart ----------
    try:
        bar_chart_file = os.path.join(app.config['STATIC_FOLDER'], 'pred_class_dist.png')
        plt.figure(figsize=(6, 4), facecolor=DARK_BG_HEX)
        ax = sns.countplot(
            x=df['Predicted Disposition'],
            order=['False Positive', 'Candidate', 'Confirmed'],
            palette=CLASS_COLORS
        )
        ax.set_title("Predicted Class Distribution", color='white')
        ax.set_xlabel("Class", color='white')
        ax.set_ylabel("Count", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        plt.tight_layout()
        plt.savefig(bar_chart_file, bbox_inches='tight', facecolor=DARK_BG_HEX)
        plt.close()
        chart_files['bar_chart'] = 'pred_class_dist.png'
    except Exception as e:
        chart_files['bar_chart'] = None

    # ---------- Pie chart ----------
    try:
        pie_chart_file = os.path.join(app.config['STATIC_FOLDER'], 'pred_class_pie.png')
        plt.figure(figsize=(5, 5), facecolor=DARK_BG_HEX)
        counts = df['Predicted Disposition'].value_counts().reindex(['False Positive', 'Candidate', 'Confirmed']).fillna(0)
        plt.pie(
            counts,
            labels=counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=CLASS_COLORS,
            textprops={'color': 'white'}
        )
        plt.title("Predicted Class Pie Chart", color='white')
        plt.savefig(pie_chart_file, bbox_inches='tight', facecolor=DARK_BG_HEX)
        plt.close()
        chart_files['pie_chart'] = 'pred_class_pie.png'
    except Exception as e:
        chart_files['pie_chart'] = None

    # ---------- Line chart: Planet radius trend ----------
    if 'koi_prad' in df.columns:
        try:
            line_chart_file = os.path.join(app.config['STATIC_FOLDER'], 'line_prad.png')
            plt.figure(figsize=(6, 4), facecolor=DARK_BG_HEX)
            plt.plot(df.index, df['koi_prad'], marker='o', linestyle='-', color='#06b6d4', linewidth=2)
            plt.title("Planet Radius Trend", color='white')
            plt.xlabel("Sample Index", color='white')
            plt.ylabel("Planet Radius (Earth radii)", color='white')
            plt.xticks(color='white')
            plt.yticks(color='white')
            plt.grid(alpha=0.2, color='#22223a')
            plt.tight_layout()
            plt.savefig(line_chart_file, bbox_inches='tight', facecolor=DARK_BG_HEX)
            plt.close()
            chart_files['line_chart'] = 'line_prad.png'
        except Exception as e:
            chart_files['line_chart'] = None
    else:
        chart_files['line_chart'] = None


    # ---------- Box plot: koi_steff per predicted class ----------
    if 'koi_steff' in df.columns:
        try:
            box_file = os.path.join(app.config['STATIC_FOLDER'], 'box_steff.png')
            plt.figure(figsize=(6, 4), facecolor=DARK_BG_HEX)
            sns.boxplot(x='Predicted Disposition', y='koi_steff', data=df, palette=CLASS_COLORS)
            plt.title("Stellar Temperature per Class", color='white')
            plt.xlabel("Predicted Class", color='white')
            plt.ylabel("Stellar Temperature (K)", color='white')
            plt.xticks(color='white')
            plt.yticks(color='white')
            plt.tight_layout()
            plt.savefig(box_file, bbox_inches='tight', facecolor=DARK_BG_HEX)
            plt.close()
            chart_files['box_chart'] = 'box_steff.png'
        except Exception as e:
            chart_files['box_chart'] = None
    else:
        chart_files['box_chart'] = None

    # ---------- Actual vs Predicted Chart ----------
    if 'koi_disposition' in df.columns:
        try:
            # Map actual labels to numbers safely (drop unknowns)
            y_true_mapped = df['koi_disposition'].map(reverse_label_map)
            # If mapping produced NaNs (unknown labels), replace with -1
            y_true = y_true_mapped.fillna(-1).astype(int)
            y_pred = preds
            # We need to ensure confusion matrix input labels are valid (exclude -1 rows)
            valid_idx = y_true != -1
            if valid_idx.sum() > 0:
                cm = confusion_matrix(y_true[valid_idx], y_pred[valid_idx], labels=[0, 1, 2])
            else:
                cm = np.zeros((3, 3), dtype=int)

            classes = ['False Positive', 'Candidate', 'Confirmed']
            actual_vs_pred_file = os.path.join(app.config['STATIC_FOLDER'], 'actual_vs_pred.png')
            plt.figure(figsize=(6, 5), facecolor=DARK_BG_HEX)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,
                        cbar=False, annot_kws={"color": "white"})
            plt.xlabel('Predicted', color='white')
            plt.ylabel('Actual', color='white')
            plt.title('Actual vs Predicted Disposition', color='white')
            plt.xticks(color='white')
            plt.yticks(color='white')
            plt.tight_layout()
            plt.savefig(actual_vs_pred_file, bbox_inches='tight', facecolor=DARK_BG_HEX)
            plt.close()
            chart_files['actual_vs_pred_chart'] = 'actual_vs_pred.png'
        except Exception as e:
            chart_files['actual_vs_pred_chart'] = None
    else:
        chart_files['actual_vs_pred_chart'] = None

    # --- Display important columns and features ---
    display_cols = []
    if 'kepid' in df.columns:
        display_cols.append('kepid')
    else:
        df.insert(0, 'Row', range(1, len(df) + 1))
        display_cols.append('Row')

    if 'koi_disposition' in df.columns:
        display_cols.append('koi_disposition')

    display_cols.append('Predicted Disposition')

    # Define significant features and their full forms
    feature_fullforms = {
        'koi_prad': 'Planet Radius (Earth radii)',
        'koi_insol': 'Incident Flux (Earth flux)',
        'koi_steff': 'Stellar Effective Temperature (K)',
        'koi_srad': 'Stellar Radius (Solar radii)',
        'koi_slogg': 'Stellar Surface Gravity (log10 cm/s²)',
        'koi_duration': 'Transit Duration (hours)',
        'koi_depth': 'Transit Depth (ppm)'
    }

    # Add existing features to display
    for feature in feature_fullforms.keys():
        if feature in df.columns:
            display_cols.append(feature)

    df_display = df[display_cols]

    # Convert the table HTML but keep it plain; styling is handled in results.html
    table_html = df_display.to_html(classes='table table-striped table-hover', index=False)

    return render_template('results.html',
                           prediction_table=table_html,
                           feature_fullforms=feature_fullforms,
                           **chart_files)


if __name__ == '__main__':
    app.run(debug=True)
