"""
BULK PREDICTION SCRIPT
======================
Use your trained model to predict labels for your 18,000 row dataset.

This script will:
1. Load your trained model
2. Load your 18k dataset
3. Predict labels for all 18k rows
4. Calculate accuracy for labels 0 and 1 (if you have some known labels)
5. Save results with predictions and confidence scores

Usage:
    python predict_18k_dataset.py
"""

import pandas as pd
import pickle
import re
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths - UPDATE THESE!
MODEL_PATH = 'best_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
INPUT_DATA_PATH = 'posts_processed2.csv'  
OUTPUT_PATH = 'predictions_output.xlsx'

# Column names in your 18k dataset - UPDATE THESE!
TEXT_COLUMN = 'cleaned_text'  # Name of column containing text to classify
TRUE_LABEL_COLUMN = None  # If you have some true labels, put column name here (e.g., 'label')

# ============================================================================
# PREPROCESSING FUNCTION (Must match training!)
# ============================================================================

def preprocess_text(text):
    """Clean text data - SAME as training"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================

def predict_bulk_dataset():
    """Predict labels for your 18k dataset"""
    
    print("\n" + "="*80)
    print("BULK PREDICTION FOR 18K DATASET")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # 1. LOAD MODEL
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading trained model and vectorizer...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print("     Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"    Error: {e}")
        print("    Make sure model.pkl and vectorizer.pkl are in the correct location!")
        return
    
    # -------------------------------------------------------------------------
    # 2. LOAD YOUR 18K DATASET
    # -------------------------------------------------------------------------
    print(f"\n[2/6] Loading dataset from: {INPUT_DATA_PATH}")
    try:
        # Try to read as Excel first
        if INPUT_DATA_PATH.endswith(('.xlsx', '.xls')):  # Adjust as needed
            df = pd.read_excel(INPUT_DATA_PATH)
        else:
            df = pd.read_csv(INPUT_DATA_PATH)
        
        print(f"     Loaded {len(df):,} rows")
        print(f"    Columns: {df.columns.tolist()}")
        
        # Check if text column exists
        if TEXT_COLUMN not in df.columns:
            print(f"     Error: Column '{TEXT_COLUMN}' not found!")
            print(f"    Available columns: {df.columns.tolist()}")
            print(f"    Update TEXT_COLUMN variable with the correct column name")
            return
            
    except FileNotFoundError:
        print(f"     Error: File not found!")
        print(f"    Update INPUT_DATA_PATH with the correct file path")
        return
    except Exception as e:
        print(f"     Error loading file: {e}")
        return
    
    # -------------------------------------------------------------------------
    # 3. PREPROCESS TEXT
    # -------------------------------------------------------------------------
    print(f"\n[3/6] Preprocessing {len(df):,} text samples...")
    df['text_processed'] = df[TEXT_COLUMN].apply(preprocess_text)
    print("     Text preprocessing complete!")
    
    # Show sample
    print(f"\n    Sample preprocessed text:")
    print(f"    Original: {df[TEXT_COLUMN].iloc[0][:100]}...")
    print(f"    Processed: {df['text_processed'].iloc[0][:100]}...")
    
    # -------------------------------------------------------------------------
    # 4. VECTORIZE AND PREDICT
    # -------------------------------------------------------------------------
    print(f"\n[4/6] Making predictions for {len(df):,} samples...")
    print("    This may take a few minutes for 18k samples...")
    
    # Vectorize
    X_vectorized = vectorizer.transform(df['text_processed'])
    print(f"     Vectorization complete! Shape: {X_vectorized.shape}")
    
    # Predict
    predictions = model.predict(X_vectorized)
    probabilities = model.predict_proba(X_vectorized)
    
    # Get confidence (max probability)
    confidence_scores = probabilities.max(axis=1)
    
    # Add to dataframe
    df['predicted_label'] = predictions
    df['confidence'] = confidence_scores
    
    # Add individual class probabilities (optional)
    for i in range(probabilities.shape[1]):
        df[f'prob_class_{i}'] = probabilities[:, i]
    
    print("     Predictions complete!")
    
    # -------------------------------------------------------------------------
    # 5. ANALYZE PREDICTIONS
    # -------------------------------------------------------------------------
    print(f"\n[5/6] Analyzing predictions...")
    
    print("\n    Predicted Label Distribution:")
    label_counts = df['predicted_label'].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"    Label {label}: {count:,} samples ({percentage:.2f}%)")
    
    print("\n    Confidence Statistics:")
    print(f"    Mean confidence: {df['confidence'].mean():.2%}")
    print(f"    Median confidence: {df['confidence'].median():.2%}")
    print(f"    Min confidence: {df['confidence'].min():.2%}")
    print(f"    Max confidence: {df['confidence'].max():.2%}")
    
    # Confidence breakdown
    print("\n    Confidence Distribution:")
    high_conf = (df['confidence'] >= 0.7).sum()
    med_conf = ((df['confidence'] >= 0.4) & (df['confidence'] < 0.7)).sum()
    low_conf = (df['confidence'] < 0.4).sum()
    print(f"    High confidence (≥70%): {high_conf:,} ({high_conf/len(df)*100:.1f}%)")
    print(f"    Medium confidence (40-70%): {med_conf:,} ({med_conf/len(df)*100:.1f}%)")
    print(f"    Low confidence (<40%): {low_conf:,} ({low_conf/len(df)*100:.1f}%)")
    
    # Focus on labels 0 and 1
    print("\n    🎯 FOCUS: Labels 0 and 1 (Your Goal)")
    label_0_count = (df['predicted_label'] == 0).sum()
    label_1_count = (df['predicted_label'] == 1).sum()
    print(f"    Label 0: {label_0_count:,} samples ({label_0_count/len(df)*100:.1f}%)")
    print(f"    Label 1: {label_1_count:,} samples ({label_1_count/len(df)*100:.1f}%)")
    
    # Average confidence for labels 0 and 1
    if label_0_count > 0:
        avg_conf_0 = df[df['predicted_label'] == 0]['confidence'].mean()
        print(f"    Label 0 avg confidence: {avg_conf_0:.2%}")
    if label_1_count > 0:
        avg_conf_1 = df[df['predicted_label'] == 1]['confidence'].mean()
        print(f"    Label 1 avg confidence: {avg_conf_1:.2%}")
    
    # If true labels are available, calculate accuracy
    if TRUE_LABEL_COLUMN and TRUE_LABEL_COLUMN in df.columns:
        print("\n    ✓ True labels found! Calculating accuracy...")
        
        # Filter to only samples with true labels
        df_labeled = df[df[TRUE_LABEL_COLUMN].notna()].copy()
        
        if len(df_labeled) > 0:
            accuracy = accuracy_score(df_labeled[TRUE_LABEL_COLUMN], 
                                     df_labeled['predicted_label'])
            print(f"\n    Overall Accuracy: {accuracy:.2%}")
            
            # Accuracy for labels 0 and 1 only
            df_0_1 = df_labeled[df_labeled[TRUE_LABEL_COLUMN].isin([0, 1])]
            if len(df_0_1) > 0:
                acc_0_1 = accuracy_score(df_0_1[TRUE_LABEL_COLUMN], 
                                        df_0_1['predicted_label'])
                print(f"    Accuracy for labels 0 and 1: {acc_0_1:.2%}")
            
            print("\n    Classification Report:")
            print(classification_report(df_labeled[TRUE_LABEL_COLUMN], 
                                       df_labeled['predicted_label']))
    
    # -------------------------------------------------------------------------
    # 6. SAVE RESULTS
    # -------------------------------------------------------------------------
    print(f"\n[6/6] Saving results...")
    
    # Save full results
    df.to_excel(OUTPUT_PATH, index=False)
    # Save as CSV instead
    # if OUTPUT_PATH.endswith('.csv'):
    #     df.to_csv(OUTPUT_PATH, index=False)
    # else:
    #     df.to_excel(OUTPUT_PATH, index=False)
    print(f"     Full results saved to: {OUTPUT_PATH}")
    
    # Save summary for labels 0 and 1
    df_0_1_predictions = df[df['predicted_label'].isin([0, 1])].copy()
    summary_path = OUTPUT_PATH.replace('.xlsx', '_labels_0_1_only.xlsx')
    df_0_1_predictions.to_excel(summary_path, index=False)
    print(f"     Labels 0 and 1 saved to: {summary_path}")
    
    # Save high confidence predictions
    df_high_conf = df[df['confidence'] >= 0.7].copy()
    high_conf_path = OUTPUT_PATH.replace('.xlsx', '_high_confidence.xlsx')
    df_high_conf.to_excel(high_conf_path, index=False)
    print(f"     High confidence predictions saved to: {high_conf_path}")
    
    print("\n" + "="*80)
    print("PREDICTION COMPLETE! ✓")
    print("="*80)
    
    print("\n Summary:")
    print(f"  • Processed: {len(df):,} samples")
    print(f"  • Predicted label 0: {label_0_count:,} ({label_0_count/len(df)*100:.1f}%)")
    print(f"  • Predicted label 1: {label_1_count:,} ({label_1_count/len(df)*100:.1f}%)")
    print(f"  • High confidence samples: {high_conf:,} ({high_conf/len(df)*100:.1f}%)")
    
    print("\n Files created:")
    print(f"  1. {OUTPUT_PATH}")
    print(f"  2. {summary_path}")
    print(f"  3. {high_conf_path}")
    
    print("\n Next Steps:")
    print("  1. Review the predictions in the Excel files")
    print("  2. Check samples with low confidence (<40%)")
    print("  3. Manually verify some predictions, especially for labels 0 and 1")
    print("  4. If accuracy is low, consider:")
    print("     - Collecting more labeled training data")
    print("     - Reviewing misclassified examples")
    print("     - Improving text preprocessing")
    
    return df


# ============================================================================
# HELPER FUNCTION: Spot Check Predictions
# ============================================================================

def spot_check_predictions(df, n_samples=10):
    """Show random sample predictions for manual review"""
    
    print("\n" + "="*80)
    print("SPOT CHECK: Random Sample Predictions")
    print("="*80)
    
    samples = df.sample(min(n_samples, len(df)))
    
    for i, (idx, row) in enumerate(samples.iterrows(), 1):
        print(f"\n--- Sample {i} ---")
        print(f"Text: {row[TEXT_COLUMN][:150]}...")
        print(f"Predicted Label: {row['predicted_label']}")
        print(f"Confidence: {row['confidence']:.2%}")
        if TRUE_LABEL_COLUMN and TRUE_LABEL_COLUMN in row:
            print(f"True Label: {row[TRUE_LABEL_COLUMN]}")
            if row['predicted_label'] == row[TRUE_LABEL_COLUMN]:
                print(" CORRECT")
            else:
                print(" INCORRECT")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print(" BEFORE RUNNING: UPDATE THESE VARIABLES!")
    print("="*80)
    print(f"\nINPUT_DATA_PATH = '{INPUT_DATA_PATH}'")
    print(f"TEXT_COLUMN = '{TEXT_COLUMN}'")
    print(f"TRUE_LABEL_COLUMN = {TRUE_LABEL_COLUMN}")
    print("\nMake sure these match your actual file and column names!")
    print("="*80)
    
    response = input("\nContinue? (y/n): ")
    
    if response.lower() == 'y':
        df_results = predict_bulk_dataset()
        
        if df_results is not None:
            # Show spot check
            response = input("\nShow random sample predictions for spot check? (y/n): ")
            if response.lower() == 'y':
                spot_check_predictions(df_results, n_samples=5)
    else:
        print("\nUpdate the variables at the top of the script and run again!")