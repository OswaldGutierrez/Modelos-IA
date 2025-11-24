# Modelos-IA

Enlace vídeo 1: https://youtu.be/daM143BC8JE


├── Machine Learning/
│   └── RandomForest/         # Carpeta que contiene el modelo Random Forest entrenado
│
├── KNN_NATICUSdroid.ipynb     # Notebook para el modelo KNN
├── RedNeuronalMulticapa_NATICUSdroid.ipynb  # Notebook para MLP
├── RegresionLogistica_NATICUSdroid.ipynb    # Notebook para Regresión Logística
├── SVM2.ipynb                 # Notebook para Máquina de Vectores de Soporte
│
├── PreProcesado_NATICUSdroid.ipynb   # Notebook de preprocesamiento de permisos Android
├── NATICUSdroid__Android_Permissions.pdf    # Documento del proyecto (teoría)
│
└── README.md                  # Descripción general del proyecto

El repositorio contiene los notebooks y recursos necesarios para entrenar y evaluar modelos de Machine Learning que analizan permisos Android y clasifican aplicaciones en benignas o maliciosas. El flujo comienza con el notebook de preprocesamiento, donde se cargan los permisos, se limpian los datos, se pasan a formato numérico binario y se genera el dataset final para entrenamiento. Ese dataset es usado por los demás notebooks, cada uno dedicado a un modelo distinto. Los archivos KNN_NATICUSdroid, RegresionLogistica_NATICUSdroid, SVM2 y RedNeuronalMulticapa_NATICUSdroid permiten ejecutar cada algoritmo de manera independiente, cargando el dataset procesado, entrenando el modelo y generando métricas como accuracy, F1-score y matriz de confusión.

Dentro de la carpeta Machine Learning/RandomForest se encuentra el modelo Random Forest ya entrenado y los archivos asociados para cargarlo y usarlo en predicciones. Este modelo suele ser el más robusto para clasificar nuevas aplicaciones, de modo que puede integrarse después en otros sistemas o probarse con nuevos permisos. Los notebooks pueden ejecutarse directamente en Google Colab o localmente con Jupyter; en ambos casos solo es necesario tener instalado Python con las librerías estándar (pandas, numpy, scikit-learn, y tensorflow si se usa la red neuronal). Cada notebook es autónomo, de modo que puedes ejecutarlos sin depender de un pipeline complejo. El PDF incluido contiene la parte conceptual del proyecto.

En los archivos COLAB ya vienen con las salidas y los resultados.
