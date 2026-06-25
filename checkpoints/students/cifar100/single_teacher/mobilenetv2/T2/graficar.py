import pandas as pd
import matplotlib.pyplot as plt

# Desactivar modo interactivo (solo guardar, no mostrar)
plt.ioff()

# Leer el CSV
df = pd.read_csv('training_log_vitbase.csv')

# Detectar nuevos entrenamientos
df['training_run'] = (df['epoch'] == 1).cumsum()

# Crear figura
plt.figure(figsize=(12, 6))

# Graficar cada entrenamiento
for run in df['training_run'].unique():
    data = df[df['training_run'] == run]
    plt.plot(data['epoch'], data['test_acc'], 
             label=f'Run {run}', linewidth=2)

# Personalizar gráfica
plt.xlabel('Época', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Comparación de Accuracy entre Entrenamientos', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Guardar la imagen
plt.savefig('comparacion_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Gráfica guardada como 'comparacion_accuracy.png'")
print(f"Se encontraron {df['training_run'].max()} entrenamientos")
print("\nResumen por entrenamiento:")
for run in df['training_run'].unique():
    data = df[df['training_run'] == run]
    print(f"Run {run}: {len(data)} épocas, Accuracy final: {data['test_acc'].iloc[-1]:.6f}")