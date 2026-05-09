"""
COMPREHENSIVE ASD DETECTION FRAMEWORK
Early Detection of Autism Spectrum Disorder using Machine Learning
Implementation with Real-World Case Studies

Author: [Your Name]
Date: 2024
Institution: [Your Institution]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1: SYNTHETIC DATASET GENERATION (Based on Your Dataset Structure)
# ============================================================================

class ASDDatasetGenerator:
    """Generate realistic ASD screening dataset based on behavioral markers"""

    def __init__(self, n_samples=300, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)

    def generate_dataset(self):
        """
        Generate dataset with:
        - A1-A10: Behavioral questionnaire responses (0-1)
        - Age: Age of child (1-6 years)
        - Gender: M/F
        - Ethnicity: Categorical
        - Family_History: Yes/No
        - AQ_Score: Autism Quotient score (0-10)
        - Class: ASD/No ASD
        - Severity: Mild/Moderate/Severe (for ASD cases)
        """

        data = {}

        # Generate behavioral indicators (A1-A10)
        # These represent: eye contact, speech patterns, repetitive behavior, 
        # social interaction, communication, etc.
        for i in range(1, 11):
            data[f'A{i}'] = np.random.randint(0, 2, self.n_samples)

        # Demographics
        data['Age'] = np.random.uniform(1, 6, self.n_samples)
        data['Gender'] = np.random.choice(['M', 'F'], self.n_samples)
        data['Ethnicity'] = np.random.choice(
            ['Asian', 'Caucasian', 'Hispanic', 'African', 'Mixed'],
            self.n_samples
        )
        data['Family_History'] = np.random.choice(['Yes', 'No'], self.n_samples, p=[0.3, 0.7])

        # Create AQ_Score based on behavioral indicators
        behavioral_sum = sum(data[f'A{i}'] for i in range(1, 11))
        # Ensure it's correctly evaluated as an int
        data['AQ_Score'] = (behavioral_sum / 10 * 10)
        data['AQ_Score'] = [int(v) for v in data['AQ_Score']]
        data['AQ_Score'] = np.clip(data['AQ_Score'], 0, 10)

        # Create Class label based on AQ_Score with some noise
        asd_threshold = 6
        data['Class'] = ['ASD' if score >= asd_threshold else 'No ASD'
                         for score in data['AQ_Score']]

        # Add family history influence
        for idx, family_hist in enumerate(data['Family_History']):
            if family_hist == 'Yes' and data['Class'][idx] == 'No ASD':
                # 40% chance of reclassification if family history
                if np.random.random() < 0.4:
                    data['Class'][idx] = 'ASD'

        # Severity classification (only for ASD cases)
        severity = []
        for idx, label in enumerate(data['Class']):
            if label == 'ASD':
                score = data['AQ_Score'][idx]
                if score < 4:
                    severity.append('Mild')
                elif score < 7:
                    severity.append('Moderate')
                else:
                    severity.append('Severe')
            else:
                severity.append('Non-ASD')
        data['Severity'] = severity

        # Create DataFrame
        df = pd.DataFrame(data)

        # Add Case_ID
        df.insert(0, 'Case_ID', [f'CASE_{i + 1:03d}' for i in range(self.n_samples)])

        return df


# ============================================================================
# SECTION 2: DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================================

class ASDDataPreprocessor:
    """Handle data preprocessing and feature engineering for ASD detection"""

    def __init__(self, df):
        self.df = df.copy()
        self.encoders = {}
        self.scaler = StandardScaler()

    def preprocess(self):
        """Complete preprocessing pipeline"""

        # Step 1: Handle missing values
        self.df.fillna(self.df.mean(numeric_only=True), inplace=True)

        # Step 2: Encode categorical variables
        categorical_cols = ['Gender', 'Ethnicity', 'Family_History', 'Class', 'Severity']

        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le

        # Step 3: Feature engineering - Create derived features
        # Composite behavioral markers
        self.df['Social_Communication_Score'] = (
                                                        self.df['A1'] + self.df['A2'] + self.df['A3']
                                                ) / 3

        self.df['Repetitive_Behavior_Score'] = (
                                                       self.df['A4'] + self.df['A5'] + self.df['A6']
                                               ) / 3

        self.df['Interaction_Score'] = (
                                               self.df['A7'] + self.df['A8'] + self.df['A9']
                                       ) / 3

        self.df['Cognitive_Score'] = self.df['A10']

        # Behavioral complexity index
        self.df['Behavioral_Complexity'] = (
                                                   self.df['Social_Communication_Score'] +
                                                   self.df['Repetitive_Behavior_Score']
                                           ) / 2

        return self.df

    @staticmethod
    def get_feature_importance_analysis():
        """Analyze which features are most important for ASD detection"""
        behavioral_features = [f'A{i}' for i in range(1, 11)]
        return behavioral_features


# ============================================================================
# SECTION 3: MACHINE LEARNING MODELS
# ============================================================================

class ASDDetectionModels:
    """Collection of ML models for ASD detection"""

    def __init__(self, x_train, x_test, y_train, y_test):
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}
        self.models = {}
        
    def _calculate_metrics(self, name, model, y_pred, y_prob):
        self.models[name] = model
        self.results[name] = {
            'predictions': y_pred,
            'probabilities': y_prob,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1': f1_score(self.y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(self.y_test, y_prob)
        }

    def train_random_forest(self):
        """
        Random Forest Classifier
        - Handles non-linear relationships in behavioral data
        - Robust to missing values
        - Provides feature importance
        """
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        self._calculate_metrics('Random Forest', model, y_pred, y_prob)
        return model

    def train_gradient_boosting(self):
        """
        Gradient Boosting Classifier
        - Sequential error correction
        - Better for complex patterns in ASD markers
        - Slower training but often higher accuracy
        """
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        self._calculate_metrics('Gradient Boosting', model, y_pred, y_prob)
        return model

    def train_svm(self):
        """
        Support Vector Machine
        - Effective for high-dimensional behavioral data
        - Good for binary classification (ASD vs Non-ASD)
        """
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        self._calculate_metrics('SVM', model, y_pred, y_prob)
        return model

    def train_logistic_regression(self):
        """
        Logistic Regression
        - Interpretable model
        - Fast training and prediction
        - Good baseline for comparison
        """
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0
        )

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        self._calculate_metrics('Logistic Regression', model, y_pred, y_prob)
        return model

    def train_adaboost(self):
        """
        AdaBoost Classifier
        - Focuses on misclassified instances
        - Good for imbalanced datasets
        """
        model = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        self._calculate_metrics('AdaBoost', model, y_pred, y_prob)
        return model

    def train_all_models(self):
        """Train all models"""
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_svm()
        self.train_logistic_regression()
        self.train_adaboost()

    def get_best_model(self):
        """Get model with highest accuracy"""
        best_model_name = max(self.results.keys(),
                              key=lambda x: self.results[x]['accuracy'])
        return best_model_name, self.models[best_model_name]

    def get_results_dataframe(self):
        """Get results as DataFrame for comparison"""
        results_df = pd.DataFrame(self.results).T
        return results_df.sort_values('accuracy', ascending=False)


# ============================================================================
# SECTION 4: MODEL EVALUATION & VISUALIZATION
# ============================================================================

class ASDModelEvaluator:
    """Evaluate and visualize model performance"""

    def __init__(self, models_dict, results_dict, x_test, y_test):
        self.models = models_dict
        self.results = results_dict
        self.X_test = x_test
        self.y_test = y_test

    def plot_model_comparison(self):
        """Compare performance metrics across models"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_names = list(self.results.keys())

        comparison_data = {}
        for metric in metrics:
            comparison_data[metric] = [
                self.results[model][metric] for model in model_names
            ]

        comparison_df = pd.DataFrame(comparison_data, index=model_names)

        fig, ax = plt.subplots(figsize=(12, 6))
        comparison_df.plot(kind='bar', ax=ax)
        ax.set_title('ASD Detection Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Models', fontsize=12)
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        return comparison_df

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (model_name, predictions) in enumerate(
                [(name, self.results[name]['predictions'])
                 for name in self.results.keys()]
        ):
            cm = confusion_matrix(self.y_test, predictions)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        # Remove extra subplots
        for idx in range(len(self.results), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))

        for model_name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            roc_auc = result['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for ASD Detection Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self, model_name, feature_names, top_n=15):
        """Plot feature importance from tree-based models"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return

        model = self.models[model_name]

        if not hasattr(model, 'feature_importances_'):
            print(f"{model_name} does not support feature importance")
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance in {model_name}',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'plots/feature_importance_{model_name.lower()}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================================
# SECTION 5: CASE STUDY ANALYSIS
# ============================================================================

class ASDCaseStudyAnalyzer:
    """Analyze real-world case studies"""

    @staticmethod
    def analyze_case_mehek():
        """
        Case Study 1: Mehek
        - Initial normal development
        - Regression after age 2
        - Speech loss, self-harming behaviors
        - Current: Severe ASD (age 10)
        """
        case = {
            'Name': 'Mehek',
            'Age': 10,
            'Diagnosis_Age': 2.5,
            'Type': 'Regression',
            'Initial_Symptoms': ['Normal early speech', 'Normal development'],
            'Current_Symptoms': ['Non-verbal', 'Self-harming (head banging, biting)',
                                 'Hyperactivity', 'Complete speech loss', 'Severe ASD'],
            'Family_History': 'No',
            'Severity': 'Severe',
            'AQ_Score': 9,
            'Intervention_Timing': 'Late (age 2.5)',
            'Behavioral_Markers': {
                'A1_Eye_Contact': 0,  # Poor
                'A2_Speech': 0,  # Absent
                'A3_Social_Interaction': 0,  # Severely impaired
                'A4_Repetitive_Behavior': 1,  # High
                'A5_Self_Stimulation': 1,  # Present
                'A6_Sensory_Issues': 1,  # Present
                'A7_Communication': 0,  # Non-verbal
                'A8_Joint_Attention': 0,  # Absent
                'A9_Pretend_Play': 0,  # Absent
                'A10_Cognitive': 0  # Significantly delayed
            }
        }
        return case

    @staticmethod
    def analyze_case_tajun():
        """
        Case Study 2: Tajun
        - Early signs from infancy
        - No speech development from start
        - Early diagnosis at 1.5 years
        - Better outcomes with early intervention
        """
        case = {
            'Name': 'Tajun',
            'Age': 7.5,
            'Diagnosis_Age': 1.5,
            'Type': 'Early-onset (no regression)',
            'Initial_Symptoms': ['Absent speech', 'Limited social interaction'],
            'Current_Symptoms': ['Limited speech ability', 'Moderate ASD',
                                 'No self-harm behaviors', 'Dietary challenges'],
            'Family_History': 'No (Twin with Tayeeba)',
            'Severity': 'Moderate',
            'AQ_Score': 6,
            'Intervention_Timing': 'Early (age 1.5)',
            'Behavioral_Markers': {
                'A1_Eye_Contact': 1,  # Minimal but present
                'A2_Speech': 0,  # Minimal
                'A3_Social_Interaction': 1,  # Impaired
                'A4_Repetitive_Behavior': 1,  # Present
                'A5_Self_Stimulation': 1,  # Present
                'A6_Sensory_Issues': 1,  # Present
                'A7_Communication': 0,  # Severely limited
                'A8_Joint_Attention': 1,  # Weak
                'A9_Pretend_Play': 1,  # Limited
                'A10_Cognitive': 1  # Moderately delayed
            }
        }
        return case

    @staticmethod
    def analyze_case_tayeeba():
        """
        Case Study 3: Tayeeba
        - Twin sister of Tajun
        - Similar early signs but better outcomes
        - Early intervention benefit
        """
        case = {
            'Name': 'Tayeeba',
            'Age': 7.5,
            'Diagnosis_Age': 1.5,
            'Type': 'Early-onset (no regression)',
            'Initial_Symptoms': ['Absent speech', 'Limited social interaction'],
            'Current_Symptoms': ['Partial speech ability', 'Mild ASD',
                                 'Better adaptive behavior', 'Dietary challenges'],
            'Family_History': 'Yes (Twin with Tajun)',
            'Severity': 'Mild',
            'AQ_Score': 4,
            'Intervention_Timing': 'Early (age 1.5)',
            'Behavioral_Markers': {
                'A1_Eye_Contact': 1,  # Present
                'A2_Speech': 1,  # Minimal but developing
                'A3_Social_Interaction': 1,  # Mildly impaired
                'A4_Repetitive_Behavior': 0,  # Less evident
                'A5_Self_Stimulation': 0,  # Minimal
                'A6_Sensory_Issues': 1,  # Present
                'A7_Communication': 1,  # Developing
                'A8_Joint_Attention': 1,  # Emerging
                'A9_Pretend_Play': 1,  # Present
                'A10_Cognitive': 1  # Mildly delayed
            }
        }
        return case

    @staticmethod
    def compare_case_studies():
        """Compare all three cases"""
        analyzer = ASDCaseStudyAnalyzer()
        cases = [
            analyzer.analyze_case_mehek(),
            analyzer.analyze_case_tajun(),
            analyzer.analyze_case_tayeeba()
        ]

        comparison_df = pd.DataFrame({
            'Name': [c['Name'] for c in cases],
            'Current_Age': [c['Age'] for c in cases],
            'Diagnosis_Age': [c['Diagnosis_Age'] for c in cases],
            'Type': [c['Type'] for c in cases],
            'Severity': [c['Severity'] for c in cases],
            'AQ_Score': [c['AQ_Score'] for c in cases],
            'Intervention_Timing': [c['Intervention_Timing'] for c in cases]
        })

        return comparison_df


# ============================================================================
# SECTION 6: MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("ASD EARLY DETECTION FRAMEWORK - COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    # Step 1: Generate Dataset
    print("\n[Step 1] Generating synthetic ASD dataset...")
    generator = ASDDatasetGenerator(n_samples=300, seed=42)
    df = generator.generate_dataset()
    print(f"✓ Dataset generated: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"\nDataset Preview:")
    print(df.head(10))

    # Step 2: Data Preprocessing
    print("\n[Step 2] Preprocessing data...")
    preprocessor = ASDDataPreprocessor(df)
    df_processed = preprocessor.preprocess()
    print(f"✓ Data preprocessing completed")
    print(f"✓ Created composite features:")
    print(f"  - Social_Communication_Score")
    print(f"  - Repetitive_Behavior_Score")
    print(f"  - Interaction_Score")
    print(f"  - Behavioral_Complexity")

    # Step 3: Prepare features and target
    print("\n[Step 3] Preparing features and target variable...")
    feature_cols = [col for col in df_processed.columns
                    if col not in ['Case_ID', 'Class', 'Severity']]
    x_features = df_processed[feature_cols]
    y_target = df_processed['Class']

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_target, test_size=0.2, random_state=42, stratify=y_target
    )

    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print(f"✓ Training set: {x_train_scaled.shape[0]} samples")
    print(f"✓ Test set: {x_test_scaled.shape[0]} samples")

    # Step 4: Train Models
    print("\n[Step 4] Training machine learning models...")
    ml_models = ASDDetectionModels(
        x_train_scaled, x_test_scaled, y_train, y_test
    )
    ml_models.train_all_models()
    print("✓ All models trained successfully")

    # Step 5: Model Evaluation
    print("\n[Step 5] Evaluating model performance...")
    results_df = ml_models.get_results_dataframe()
    print("\nModel Performance Comparison:")
    print(results_df.round(4))

    best_model_name, best_model = ml_models.get_best_model()
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"  Accuracy: {ml_models.results[best_model_name]['accuracy']:.4f}")
    print(f"  Precision: {ml_models.results[best_model_name]['precision']:.4f}")
    print(f"  Recall: {ml_models.results[best_model_name]['recall']:.4f}")
    print(f"  F1-Score: {ml_models.results[best_model_name]['f1']:.4f}")
    print(f"  ROC-AUC: {ml_models.results[best_model_name]['roc_auc']:.4f}")

    # Step 6: Visualizations
    print("\n[Step 6] Generating visualizations...")
    evaluator = ASDModelEvaluator(
        ml_models.models, ml_models.results, x_test_scaled, y_test
    )

    print("  - Plotting model comparison...")
    evaluator.plot_model_comparison()

    print("  - Plotting confusion matrices...")
    evaluator.plot_confusion_matrices()

    print("  - Plotting ROC curves...")
    evaluator.plot_roc_curves()

    # Feature importance for tree-based models
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        print("  - Plotting feature importance...")
        evaluator.plot_feature_importance(best_model_name, feature_cols, top_n=15)

    # Step 7: Case Study Analysis
    print("\n[Step 7] Analyzing real-world case studies...")
    case_analyzer = ASDCaseStudyAnalyzer()

    print("\nCase Study Comparison:")
    comparison = case_analyzer.compare_case_studies()
    print(comparison.to_string(index=False))

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return {
        'dataset': df,
        'dataset_processed': df_processed,
        'models': ml_models,
        'results': results_df,
        'best_model': (best_model_name, best_model),
        'evaluator': evaluator,
        'cases': comparison
    }


if __name__ == "__main__":
    results = main()