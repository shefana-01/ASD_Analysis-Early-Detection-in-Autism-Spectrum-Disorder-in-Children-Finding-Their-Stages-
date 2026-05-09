from .explainability import ModelExplainer
from .fairness import FairnessAnalyzer
from .validation import ModelValidator
from .uncertainty import UncertaintyQuantifier

import numpy as np
try:
    import shap
except ImportError:
    shap = None


class ExplainableASDModel:
    """Make ML predictions interpretable for clinicians"""

    @staticmethod
    def generate_shap_explanation(model, X_test, feature_names, case_idx):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Get explanation for single case
        case_explanation = {
            'predicted_probability': model.predict_proba(X_test[case_idx:case_idx + 1])[0][1],
            'contributing_features': [
                {
                    'feature': feature_names[i],
                    'value': X_test[case_idx, i],
                    'impact': shap_values[1][case_idx, i]  # Positive = supports ASD
                }
                for i in range(len(feature_names))
            ]
        }

        # Sort by impact magnitude
        case_explanation['contributing_features'].sort(
            key=lambda x: abs(x['impact']), reverse=True
        )

        return case_explanation

    @staticmethod
    def clinical_report(explanation, feature_names=None):
        report = "ASD RISK ASSESSMENT REPORT\n"
        report += "=" * 50 + "\n\n"

        prob = explanation['predicted_probability']
        report += f"ASD Probability: {prob * 100:.1f}%\n"

        if prob > 0.7:
            report += "Risk Level: HIGH - Recommend comprehensive evaluation\n\n"
        elif prob > 0.5:
            report += "Risk Level: MODERATE - Close monitoring recommended\n\n"
        else:
            report += "Risk Level: LOW - Standard developmental surveillance\n\n"

        report += "Key Contributing Factors:\n"
        for i, feature in enumerate(explanation['contributing_features'][:5], 1):
            direction = "↑ increases risk" if feature['impact'] > 0 else "↓ decreases risk"
            report += f"{i}. {feature['feature']}: {direction}\n"

        return report


class PredictionUncertainty:
    """Quantify confidence in predictions"""

    @staticmethod
    def confidence_intervals(model, X_test, confidence=0.95):
        bootstrap_predictions = []
        for _ in range(100):
            indices = np.random.choice(len(X_test), len(X_test), replace=True)
            X_boot = X_test[indices]
            probs = model.predict_proba(X_boot)[:, 1]
            bootstrap_predictions.append(probs)

        lower = np.percentile(bootstrap_predictions, (1 - confidence) / 2 * 100, axis=0)
        upper = np.percentile(bootstrap_predictions, (1 + confidence) / 2 * 100, axis=0)

        return lower, upper

    @staticmethod
    def identify_difficult_cases(model, X_test):
        probabilities = model.predict_proba(X_test)
        confidence = 1 - np.abs(probabilities[:, 1] - 0.5) * 2
        difficult_indices = np.where(confidence < 0.7)[0]

        return difficult_indices, confidence[difficult_indices]


class FairnessAnalysis:
    """Ensure model works equally across demographic groups"""

    @staticmethod
    def demographic_parity(predictions, y_true, demographics, demographic_groups):
        fairness_metrics = {}

        for group in demographic_groups:
            mask = demographics == group
            group_predictions = predictions[mask]
            group_true = y_true[mask]

            if len(group_true) == 0:
                continue

            detection_rate = np.mean(group_predictions == 1)
            true_negatives = np.sum(group_true == 0)
            true_positives = np.sum(group_true == 1)

            false_positive_rate = np.mean(
                (group_predictions == 1) & (group_true == 0)) / true_negatives if true_negatives > 0 else 0
            false_negative_rate = np.mean(
                (group_predictions == 0) & (group_true == 1)) / true_positives if true_positives > 0 else 0

            fairness_metrics[group] = {
                'detection_rate': detection_rate,
                'fpr': false_positive_rate,
                'fnr': false_negative_rate
            }

        if fairness_metrics:
            detection_rates = [fairness_metrics[g]['detection_rate'] for g in fairness_metrics]
            max_disparity = max(detection_rates) - min(detection_rates)
        else:
            max_disparity = 0

        return fairness_metrics, max_disparity

    @staticmethod
    def intersectionality_analysis(predictions, y_true, gender, ethnicity):
        combinations = np.unique(list(zip(gender, ethnicity)), axis=0)

        results = {}
        for combo in combinations:
            mask = (gender == combo[0]) & (ethnicity == combo[1])
            if np.sum(mask) > 0:
                results[tuple(combo)] = {
                    'count': np.sum(mask),
                    'accuracy': np.mean(predictions[mask] == y_true[mask]),
                    'asd_prevalence': np.mean(y_true[mask])
                }

        return results