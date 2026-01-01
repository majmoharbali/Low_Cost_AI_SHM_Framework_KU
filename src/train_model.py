"""
train_model.py

Pipeline: Physics-Informed Model Training
-----------------------------------------
1. Generates synthetic training data using the Physics Engine.
2. Extracts features (PSD, Statistical Moments).
3. Trains a Calibrated Random Forest Classifier.
4. Optimizes decision thresholds using Cost-Sensitive ROC Analysis.

Output: artifacts/shm_deployment_package_enhanced.pkl
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis

# Import Machine Learning Libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

# Fix path to allow importing from core/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.physics_engine import ModalDamageSimulator

# --- CONFIGURATION ---
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'shm_deployment_package_enhanced.pkl')

class DataGeneratorWrapper:
    """Wrapper to generate large datasets using the ModalDamageSimulator."""
    def __init__(self, n_samples=4500):
        self.sim = ModalDamageSimulator(n_modes=3)
        self.n_samples = n_samples
        self.t = np.linspace(0, 10, 1000, endpoint=False) # 10 seconds at 100Hz

    def generate_dataset(self):
        X, y = [], []
        samples_per_class = self.n_samples // 3
        print(f"[Training] Generating {self.n_samples} physics-based samples...")

        # 1. Healthy
        self.sim.apply_damage(0.0)
        for _ in range(samples_per_class):
            # Simulate random sensor read from the beam
            resp = self.sim.generate_response(self.t)
            X.append(resp[0] + np.random.normal(0, 0.05, len(self.t))) # Use Sensor 0
            y.append(0)

        # 2. Incipient (10-30% Stiffness Loss)
        for _ in range(samples_per_class):
            self.sim.apply_damage(np.random.uniform(0.1, 0.3))
            resp = self.sim.generate_response(self.t)
            X.append(resp[0] + np.random.normal(0, 0.05, len(self.t)))
            y.append(1)

        # 3. Severe (40-80% Stiffness Loss)
        for _ in range(samples_per_class):
            self.sim.apply_damage(np.random.uniform(0.4, 0.8))
            resp = self.sim.generate_response(self.t)
            X.append(resp[0] + np.random.normal(0, 0.05, len(self.t)))
            y.append(2)

        return np.array(X), np.array(y)

class FeatureExtractor:
    """Extracts Physics-Domain Features."""
    def extract(self, X: np.ndarray, fs=100) -> pd.DataFrame:
        features = []
        for sig in X:
            f, Pxx = signal.welch(sig, fs, nperseg=256)
            rms = np.sqrt(np.mean(sig**2))
            features.append({
                'mean': np.mean(sig), 
                'std_dev': np.std(sig), 
                'rms': rms,
                'peak': np.max(np.abs(sig)), 
                'skewness': skew(sig), 
                'kurtosis': kurtosis(sig),
                'crest_factor': np.max(np.abs(sig)) / (rms + 1e-9),
                'dominant_freq': f[np.argmax(Pxx)],
                'spectral_centroid': np.sum(f * Pxx) / (np.sum(Pxx) + 1e-9)
            })
        return pd.DataFrame(features)

class StatisticalThresholdCalculator:
    """Optimizes decision boundaries using Cost-Sensitive ROC Analysis."""
    def __init__(self, cost_fp=1000, cost_fn=50000):
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn

    def calculate(self, y_true, y_probs):
        thresholds = {}
        for cls, name in [(1, 'incipient'), (2, 'severe')]:
            y_bin = (y_true == cls).astype(int)
            probs = y_probs[:, cls]
            fpr, tpr, thresh = roc_curve(y_bin, probs)
            # Minimize Expected Cost
            costs = (self.cost_fp * fpr) + (self.cost_fn * (1 - tpr))
            best_idx = np.argmin(costs)
            thresholds[f'{name}_threshold'] = float(thresh[best_idx])
            print(f"  > Optimized {name.title()} Threshold: {thresh[best_idx]:.4f} (Min Cost: ${costs[best_idx]:,.0f})")
        return thresholds

def main():
    # 1. Generate Data
    gen = DataGeneratorWrapper()
    X_raw, y = gen.generate_dataset()
    
    # 2. Extract Features
    extractor = FeatureExtractor()
    X_feats = extractor.extract(X_raw)
    
    # 3. Train Model
    print("[Training] Training Calibrated Random Forest...")
    base_rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    model = CalibratedClassifierCV(base_rf, method='isotonic', cv=3)
    
    # Explainer model (SHAP needs standard RF)
    explainer_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    scaler = StandardScaler()
    selector = SelectKBest(f_classif, k=8)
    
    # Pipeline
    X_scaled = scaler.fit_transform(X_feats)
    X_sel = selector.fit_transform(X_scaled, y)
    
    model.fit(X_sel, y)
    explainer_rf.fit(X_sel, y)
    
    # 4. Optimize Thresholds
    probs = model.predict_proba(X_sel)
    calc = StatisticalThresholdCalculator()
    thresholds = calc.calculate(y, probs)
    
    # 5. Save Artifacts
    package = {
        'model': model,
        'scaler': scaler,
        'feature_selector': selector,
        'explainer_model': explainer_rf,
        'confidence_stats': thresholds
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(package, f)
    print(f"[Success] Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
