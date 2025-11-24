# Análisis de Malware en Android usando Random Forest


# IMPORTACIONES DE LIBRERÍAS
import sys
import time
from datetime import datetime, timedelta

print("\nCargando dependencias,,,\n")
start_import = time.time()

try:
    import pandas as pd
    print("pandas")
    import numpy as np
    print("numpy")
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, roc_auc_score, confusion_matrix,
                                classification_report)
    print("sklearn")
    from collections import Counter
    print("collections")
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("matplotlib/seaborn")
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


# ============================================================================
# MEDIDOR DEL PROGRESO
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
        
        # Actualizar cada 10 fits o cada 5 segundos
        if self.completed_fits % 10 == 0 or (current_time - self.last_update) > 5:
            elapsed = current_time - self.start_time
            progress = (self.completed_fits / self.total_fits) * 100
            
            # Estimar tiempo restante
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
# 1. CARGA DE DATOS
# ============================================================================
print("\n" + "="*80)
print("CARGANDO DATASET")
print("="*80)

try:
    d = pd.read_csv("data.csv")
    print(f"Dataset: {d.shape[0]:,} instancias, {d.shape[1]-1} características")
except FileNotFoundError:
    print("ERROR: 'data.csv' no encontrado")
    sys.exit(1)

X = d.drop('Result', axis=1)
y = d['Result']

# ============================================================================
# 2. ANÁLISIS DE DESBALANCE
# ============================================================================
print("\n" + "="*80)
print("ANÁLISIS DE DESBALANCE")
print("="*80)

class_distribution = Counter(y)
minority = min(class_distribution.values())
majority = max(class_distribution.values())
imbalance_ratio = majority / minority
use_balanced = imbalance_ratio > 1.5

for clase, count in class_distribution.items():
    percentage = (count / len(y)) * 100
    clase_name = "Malware" if clase == 1 else "Benigna"
    print(f"  Clase {clase} ({clase_name}): {count:,} ({percentage:.2f}%)")

print(f"  Ratio: {imbalance_ratio:.2f}:1 → {'Balanced' if use_balanced else 'Normal'}")

# ============================================================================
# 3. VALIDACIÓN CRUZADA
# ============================================================================
print("\n" + "="*80)
print("CONFIGURACIÓN")
print("="*80)

cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
print("10-Fold Stratified CV")

# ============================================================================
# 4. HIPERPARÁMETROS
# ============================================================================
param_grid = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [20, 25, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'class_weight': ['balanced'] if use_balanced else ['balanced']
}

total_combinations = np.prod([len(v) for v in param_grid.values()])
total_fits = total_combinations * 10

print(f"\nHiperparámetros:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

print(f"\nTotal:")
print(f"  • Combinaciones: {total_combinations}")
print(f"  • Entrenamientos: {total_fits:,}")

# ============================================================================
# 5. MÉTRICAS
# ============================================================================
scoring_metrics = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

print("\nMétricas: Accuracy, Precision, Recall, F1, ROC-AUC")

# ============================================================================
# 6. GRID SEARCH CON PROGRESO
# ============================================================================
print("\n" + "="*80)
print("ENTRENAMIENTO")
print("="*80)
print(f"   Procesando {total_combinations} combinaciones × 10 folds...\n")

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
try:
    grid_search.fit(X, y)
    print()
except KeyboardInterrupt:
    print("\n\nInterrumpido")
    sys.exit(0)

grid_elapsed = time.time() - grid_start_time
grid_elapsed_str = str(timedelta(seconds=int(grid_elapsed)))

print("="*80)
print(f"\nGrid Search completado!")
print(f"  Tiempo: {grid_elapsed/60:.2f} min ({grid_elapsed_str})")
print(f"  Velocidad: {total_fits/grid_elapsed:.2f} fits/seg")
print(f"  Mejor F1-Score: {grid_search.best_score_:.4f}")

print(f"\nMejores hiperparámetros:")
print("-" * 60)
for param, value in sorted(grid_search.best_params_.items()):
    print(f"  {param:<25}: {value}")
print("-" * 60)

print(f"\nTop 3 configuraciones:")
cv_results_df = pd.DataFrame(grid_search.cv_results_)
top_3 = cv_results_df.nlargest(3, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
for idx, row in enumerate(top_3.itertuples(), 1):
    print(f"#{idx} F1={row.mean_test_score:.4f} (±{row.std_test_score:.4f})")

# ============================================================================
# 7. EVALUACIÓN COMPLETA
# ============================================================================
print("\n" + "="*80)
print("EVALUACIÓN FINAL")
print("="*80)

best_model = grid_search.best_estimator_

print("\nValidación cruzada completa...")
cv_results = cross_validate(
    best_model, X, y,
    cv=cv_strategy,
    scoring=scoring_metrics,
    return_train_score=True,
    n_jobs=-1
)

print("\nRESULTADOS:")
print("-" * 60)
for metric_name in scoring_metrics.keys():
    test_scores = cv_results[f'test_{metric_name}']
    train_scores = cv_results[f'train_{metric_name}']
    gap = train_scores.mean() - test_scores.mean()
    
    print(f"{metric_name.upper():<12} Test: {test_scores.mean():.4f}±{test_scores.std():.4f}  "
          f"Train: {train_scores.mean():.4f}  Gap: {gap:.4f}")

# ============================================================================
# 8. OVERFITTING
# ============================================================================
print("\n" + "="*80)
print("OVERFITTING")
print("="*80)

for metric_name in scoring_metrics.keys():
    test_mean = cv_results[f'test_{metric_name}'].mean()
    train_mean = cv_results[f'train_{metric_name}'].mean()
    gap = train_mean - test_mean
    gap_pct = (gap / train_mean) * 100 if train_mean != 0 else 0
    
    status = "Alto" if gap_pct > 15 else "Moderado" if gap_pct > 10 else "ℹLeve" if gap_pct > 5 else "Bien"
    print(f"{metric_name.upper():<12} Gap: {gap_pct:>5.2f}%  {status}")

# ============================================================================
# 9. IMPORTANCIA
# ============================================================================
print("\n" + "="*80)
print("IMPORTANCIA DE CARACTERÍSTICAS")
print("="*80)

print("\nCalculando importancia...")
best_model.fit(X, y)
feature_importance = pd.DataFrame({
    'Permiso': X.columns,
    'Importancia': best_model.feature_importances_
}).sort_values('Importancia', ascending=False)

print("\nTOP 15 PERMISOS:")
print("-"*80)
cumulative = 0
for idx, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
    cumulative += row['Importancia']
    print(f"{idx:>2}. {row['Permiso']:<55} {row['Importancia']:.6f} ({cumulative*100:.1f}%)")

print("-"*80)

# ============================================================================
# 10. VISUALIZACIONES
# ============================================================================
print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES")
print("="*80)

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ----------------------------------------------------------------------------
# GRAFICO 1: IMPORTANCIA DE CARACTERÍSTICAS (TOP 20)
# ----------------------------------------------------------------------------
print("\nGenerando grafico de importancia de caracteristicas...")

fig, ax = plt.subplots(figsize=(12, 8))
top_20 = feature_importance.head(20)

# Crear barplot horizontal
colors = plt.cm.viridis(np.linspace(0.3, 0.9, 20))
bars = ax.barh(range(len(top_20)), top_20['Importancia'], color=colors)

# Configurar ejes
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels([p.replace('android.permission.', '').replace('com.', '') 
                     for p in top_20['Permiso']], fontsize=10)
ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
ax.set_ylabel('Permiso', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Permisos más Importantes para Detección de Malware', 
             fontsize=14, fontweight='bold', pad=20)

# Agregar valores en las barras
for i, (bar, val) in enumerate(zip(bars, top_20['Importancia'])):
    ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)

# Invertir eje Y para que el más importante esté arriba
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("Guardado: feature_importance.png")
plt.close()

# ----------------------------------------------------------------------------
# GRAFICO 2: MATRIZ DE CONFUSIÓN
# ----------------------------------------------------------------------------
print("\nGenerando matriz de confusion...")

# Predecir en todo el conjunto (para visualización)
y_pred = best_model.predict(X)
cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))

# Crear heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            square=True, linewidths=2, linecolor='white',
            cbar_kws={'label': 'Cantidad'}, ax=ax,
            annot_kws={'size': 14, 'weight': 'bold'})

# Configurar etiquetas
ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
ax.set_ylabel('Real', fontsize=12, fontweight='bold')
ax.set_title('Matriz de Confusión - Random Forest\n(Conjunto Completo)', 
             fontsize=14, fontweight='bold', pad=20)

# Etiquetas de clases
ax.set_xticklabels(['Benigna (0)', 'Malware (1)'], fontsize=11)
ax.set_yticklabels(['Benigna (0)', 'Malware (1)'], fontsize=11, rotation=0)

# Calcular métricas
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Agregar texto con métricas
metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Guardado: confusion_matrix.png")
plt.close()

# ----------------------------------------------------------------------------
# GRAFICO 3: COMPARACIÓN TRAIN VS TEST
# ----------------------------------------------------------------------------
print("\nGenerando comparacion de metricas Train vs Test...")

fig, ax = plt.subplots(figsize=(12, 7))

# Preparar datos
metrics_names = list(scoring_metrics.keys())
train_means = [cv_results[f'train_{m}'].mean() for m in metrics_names]
test_means = [cv_results[f'test_{m}'].mean() for m in metrics_names]
test_stds = [cv_results[f'test_{m}'].std() for m in metrics_names]

x = np.arange(len(metrics_names))
width = 0.35

# Crear barras
bars1 = ax.bar(x - width/2, train_means, width, label='Train',
               color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, test_means, width, label='Test',
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5,
               yerr=test_stds, capsize=5)

# Configurar ejes
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Métrica', fontsize=12, fontweight='bold')
ax.set_title('Comparación de Métricas: Train vs Test (10-Fold CV)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in metrics_names], fontsize=11)
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.05])

# Agregar valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Agregar línea de referencia
ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1, alpha=0.5, label='95%')

plt.tight_layout()
plt.savefig('train_vs_test_comparison.png', dpi=300, bbox_inches='tight')
print("Guardado: train_vs_test_comparison.png")
plt.close()

# ----------------------------------------------------------------------------
# GRAFICO 4: GAP DE OVERFITTING
# ----------------------------------------------------------------------------
print("\nGenerando analisis de overfitting...")

fig, ax = plt.subplots(figsize=(10, 6))

# Calcular gaps
gaps = []
gap_percentages = []
for m in metrics_names:
    train_mean = cv_results[f'train_{m}'].mean()
    test_mean = cv_results[f'test_{m}'].mean()
    gap = train_mean - test_mean
    gap_pct = (gap / train_mean) * 100 if train_mean != 0 else 0
    gaps.append(gap)
    gap_percentages.append(gap_pct)

# Crear barplot con colores según severidad
colors = []
for gap_pct in gap_percentages:
    if gap_pct > 15:
        colors.append('#e74c3c')  # Rojo - Alto
    elif gap_pct > 10:
        colors.append('#f39c12')  # Naranja - Moderado
    elif gap_pct > 5:
        colors.append('#f1c40f')  # Amarillo - Leve
    else:
        colors.append('#2ecc71')  # Verde - Bien

bars = ax.bar(metrics_names, gap_percentages, color=colors, alpha=0.8,
              edgecolor='black', linewidth=1.5)

# Configurar ejes
ax.set_ylabel('Gap (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Métrica', fontsize=12, fontweight='bold')
ax.set_title('Análisis de Overfitting por Métrica\n(Gap entre Train y Test)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticklabels([m.upper() for m in metrics_names], fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Agregar valores
for bar, gap_pct in zip(bars, gap_percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{gap_pct:.2f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Agregar líneas de referencia
ax.axhline(y=5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Leve (5%)')
ax.axhline(y=10, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderado (10%)')
ax.axhline(y=15, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Alto (15%)')
ax.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
print("Guardado: overfitting_analysis.png")
plt.close()

print("\n" + "="*80)
print("VISUALIZACIONES COMPLETADAS")
print("="*80)
print("\nArchivos generados:")
print("  feature_importance.png - Top 20 permisos mas importantes")
print("  confusion_matrix.png - Matriz de confusion del modelo")
print("  train_vs_test_comparison.png - Comparacion de metricas")
print("  overfitting_analysis.png - Analisis de gaps por metrica")
print("="*80)

