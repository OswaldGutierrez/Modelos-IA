# Análisis de Malware en Android usando Random Forest con PCA y UMAP

import sys
import time
from datetime import timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PROGRESS TRACKER
# ============================================================================
class ProgressTracker:
    def __init__(self, total_fits):
        self.total_fits = total_fits
        self.completed_fits = 0
        self.start_time = time.time()
        self.last_update = time.time()
        
    def update(self):
        self.completed_fits += 1
        current_time = time.time()
        
        if self.completed_fits % 10 == 0 or (current_time - self.last_update) > 5:
            elapsed = current_time - self.start_time
            progress = (self.completed_fits / self.total_fits) * 100
            
            if self.completed_fits > 0:
                time_per_fit = elapsed / self.completed_fits
                remaining_fits = self.total_fits - self.completed_fits
                eta_seconds = time_per_fit * remaining_fits
                eta_str = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta_str = "calculando..."
            
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            print(f"\rProgreso: {self.completed_fits}/{self.total_fits} ({progress:.1f}%) | "
                  f"Tiempo: {elapsed_str} | ETA: {eta_str}     ", end='', flush=True)
            
            self.last_update = current_time

progress_tracker = None

# ============================================================================
# CARGA DE DATOS
# ============================================================================
print("\n" + "="*80)
print("CARGANDO DATASET")
print("="*80)

d = pd.read_csv("data.csv")
print(f"Dataset: {d.shape[0]:,} instancias, {d.shape[1]-1} características")

X = d.drop('Result', axis=1)
y = d['Result']

# ============================================================================
# ANÁLISIS DE DESBALANCE
# ============================================================================
print("\n" + "="*80)
print("ANÁLISIS DE DESBALANCE")
print("="*80)

class_distribution = Counter(y)
minority = min(class_distribution.values())
majority = max(class_distribution.values())
imbalance_ratio = majority / minority

for clase, count in class_distribution.items():
    percentage = (count / len(y)) * 100
    clase_name = "Malware" if clase == 1 else "Benigna"
    print(f"  Clase {clase} ({clase_name}): {count:,} ({percentage:.2f}%)")

print(f"  Ratio: {imbalance_ratio:.2f}:1")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
print("\n" + "="*80)
print("CONFIGURACIÓN")
print("="*80)

cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
print("10-Fold Stratified CV")

param_grid = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [20, 25, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'class_weight': ['balanced']
}

total_combinations = np.prod([len(v) for v in param_grid.values()])
total_fits = total_combinations * 10

print(f"\nTotal: {total_combinations} combinaciones × 10 folds = {total_fits:,} entrenamientos")

scoring_metrics = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# ============================================================================
# GRID SEARCH
# ============================================================================
print("\n" + "="*80)
print("ENTRENAMIENTO")
print("="*80)

grid_start_time = time.time()
progress_tracker = ProgressTracker(total_fits)

class ProgressRandomForest(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None):
        result = super().fit(X, y, sample_weight=sample_weight)
        if progress_tracker:
            progress_tracker.update()
        return result

grid_search = GridSearchCV(
    estimator=ProgressRandomForest(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='f1',
    n_jobs=1,
    verbose=0,
    return_train_score=True
)

print("="*80)
grid_search.fit(X, y)
print()

grid_elapsed = time.time() - grid_start_time
print("="*80)
print(f"\nGrid Search completado en {grid_elapsed/60:.2f} min")
print(f"Mejor F1-Score: {grid_search.best_score_:.4f}")

print(f"\nMejores hiperparámetros:")
for param, value in sorted(grid_search.best_params_.items()):
    print(f"  {param:<25}: {value}")

# ============================================================================
# EVALUACIÓN SIN REDUCCIÓN
# ============================================================================
print("\n" + "="*80)
print("EVALUACIÓN: MODELO ORIGINAL")
print("="*80)

best_model = grid_search.best_estimator_

cv_results = cross_validate(
    best_model, X, y,
    cv=cv_strategy,
    scoring=scoring_metrics,
    return_train_score=True,
    n_jobs=-1
)

print("\nRESULTADOS:")
for metric_name in scoring_metrics.keys():
    test_scores = cv_results[f'test_{metric_name}']
    print(f"{metric_name.upper():<12} {test_scores.mean():.4f} ± {test_scores.std():.4f}")

# ============================================================================
# PCA
# ============================================================================
print("\n" + "="*80)
print("ANÁLISIS PCA")
print("="*80)

print("\nCRITERIO: Retener 95% de la varianza")
print("JUSTIFICACIÓN: Balance óptimo entre reducción y preservación de información")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

variance_cumsum = np.cumsum(pca_full.explained_variance_ratio_)
threshold = 0.95
n_components_pca = np.argmax(variance_cumsum >= threshold) + 1

print(f"\nComponentes: {X.shape[1]} → {n_components_pca}")
print(f"Reducción: {((X.shape[1] - n_components_pca) / X.shape[1] * 100):.2f}%")
print(f"Varianza explicada: {variance_cumsum[n_components_pca-1]*100:.2f}%")

pca = PCA(n_components=n_components_pca, random_state=42)
X_pca = pca.fit_transform(X_scaled)

model_pca = RandomForestClassifier(**grid_search.best_params_, random_state=42, n_jobs=-1)
cv_results_pca = cross_validate(
    model_pca, X_pca, y,
    cv=cv_strategy,
    scoring=scoring_metrics,
    return_train_score=True,
    n_jobs=-1
)

print("\nRESULTADOS CON PCA:")
for metric_name in scoring_metrics.keys():
    test_scores = cv_results_pca[f'test_{metric_name}']
    print(f"{metric_name.upper():<12} {test_scores.mean():.4f} ± {test_scores.std():.4f}")

# ============================================================================
# UMAP
# ============================================================================
print("\n" + "="*80)
print("ANÁLISIS UMAP")
print("="*80)

print("\nCRITERIO: n_components = √(n_features)")
print("JUSTIFICACIÓN: Heurística que balancea expresividad y reducción dimensional")

n_components_umap = int(np.sqrt(X.shape[1]))
print(f"\nComponentes: {X.shape[1]} → {n_components_umap}")
print(f"Reducción: {((X.shape[1] - n_components_umap) / X.shape[1] * 100):.2f}%")

print("\nAplicando UMAP...")
umap_reducer = umap.UMAP(
    n_components=n_components_umap,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
    verbose=False
)

X_umap = umap_reducer.fit_transform(X_scaled)

model_umap = RandomForestClassifier(**grid_search.best_params_, random_state=42, n_jobs=-1)
cv_results_umap = cross_validate(
    model_umap, X_umap, y,
    cv=cv_strategy,
    scoring=scoring_metrics,
    return_train_score=True,
    n_jobs=-1
)

print("\nRESULTADOS CON UMAP:")
for metric_name in scoring_metrics.keys():
    test_scores = cv_results_umap[f'test_{metric_name}']
    print(f"{metric_name.upper():<12} {test_scores.mean():.4f} ± {test_scores.std():.4f}")

# ============================================================================
# TABLA COMPARATIVA
# ============================================================================
print("\n" + "="*80)
print("TABLA COMPARATIVA")
print("="*80)

comparison_data = []

# Original
row = {
    'Modelo': 'Original',
    'Componentes': X.shape[1],
    'Reducción (%)': 0.0
}
for metric_name in scoring_metrics.keys():
    row[metric_name.capitalize()] = f"{cv_results[f'test_{metric_name}'].mean():.4f} ± {cv_results[f'test_{metric_name}'].std():.4f}"
comparison_data.append(row)

# PCA
row = {
    'Modelo': 'PCA (95%)',
    'Componentes': n_components_pca,
    'Reducción (%)': round(((X.shape[1] - n_components_pca) / X.shape[1] * 100), 2)
}
for metric_name in scoring_metrics.keys():
    row[metric_name.capitalize()] = f"{cv_results_pca[f'test_{metric_name}'].mean():.4f} ± {cv_results_pca[f'test_{metric_name}'].std():.4f}"
comparison_data.append(row)

# UMAP
row = {
    'Modelo': 'UMAP (√n)',
    'Componentes': n_components_umap,
    'Reducción (%)': round(((X.shape[1] - n_components_umap) / X.shape[1] * 100), 2)
}
for metric_name in scoring_metrics.keys():
    row[metric_name.capitalize()] = f"{cv_results_umap[f'test_{metric_name}'].mean():.4f} ± {cv_results_umap[f'test_{metric_name}'].std():.4f}"
comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)