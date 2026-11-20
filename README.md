================================================================================
F1 DIRTY AIR & SLIPSTREAM TELEMETRY ANALYZER (HPC + YOLOv8 + FastF1)
================================================================================
Sistema distribuido de visión por computadora y fusión multimodal de telemetría 
para cuantificar el impacto aerodinámico (Dirty Air) en Fórmula 1.

- Autor: Ángel Jored Cruz Flores (AngelCruz-DS)
- Materia: Cómputo de Alto Desempeño (HPC) — ESCOM / IPN
- Infraestructura: Clúster CIC-IPN (ibh1 / Nodos GPU ibng2, ibng3)

--------------------------------------------------------------------------------
1. DESCRIPCIÓN Y MOTIVACIÓN DEL PROYECTO
--------------------------------------------------------------------------------
En la Fórmula 1 moderna, cuando un monoplaza persigue a otro a menos de 1.5 
segundos, experimenta dos fenómenos aerodinámicos opuestos:

1) En recta (Rebufo / Slipstream): Reducción de resistencia al avance (drag), 
   lo que incrementa la velocidad punta.
2) En curva rápida (Aire Sucio / Dirty Air): Pérdida severa de carga 
   aerodinámica (downforce) en el alerón delantero y suelo, provocando 
   subviraje, degradación térmica de neumáticos y pérdida de tiempo por sector.

EL RETO DE INGENIERÍA:
Aunque los equipos cuentan con telemetría interna de su propio vehículo 
(velocidad, acelerador, freno, marchas, DRS), no tienen sensores físicos 
instalados en los autos rivales.

NUESTRA SOLUCIÓN:
Construir un pipeline de Cómputo de Alto Desempeño (HPC) que procesa en 
paralelo transmisiones de video onboard (cámara T-Cam) mediante YOLOv8 para 
detectar monoplazas rivales, estimar su distancia física en metros cuadro por 
cuadro y fusionar esa serie de tiempo visual con la telemetría oficial real de 
la FIA (FastF1). Todo el sistema culmina en un Dashboard Web Interactivo 
(Pit Wall Telemetry) para inspección visual.

--------------------------------------------------------------------------------
2. ARQUITECTURA DEL SISTEMA Y ESTRATEGIA HPC
--------------------------------------------------------------------------------
El proyecto está diseñado para aprovechar al máximo la topología real del 
clúster del CIC (148.204.63.158), separando la orquestación del cómputo pesado:

+-------------------------------------------------------------------------+
|                        FASE 1: INGESTA MULTIMODAL                       |
|  [FastF1 API] -> Telemetría Real (Speed, Throttle, Brake, DRS, X, Y, Z) |
|  [Video Onboard] -> Flujo MP4 1080p @ 50/60 FPS (T-Cam Perseguidor)      |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|               FASE 2: CLÚSTER CIC-IPN (SLURM + MULTI-GPU)               |
|  Nodo Maestro (ibh1 - Rocky Linux 9.7):                                 |
|   ├── Fragmentación temporal del video en N chunks equilibrados         |
|   └── Despacho de trabajos vía SLURM (sbatch) a partición 'gpu'         |
|                                                                         |
|  Nodos de Cómputo (ibng2 & ibng3 - 56 CPUs, 256 GB RAM, 8 GPUs K80):    |
|   ├── Worker 0..3 (ibng2) -> Inferencia YOLOv8 en paralelo por chunk    |
|   └── Worker 4..7 (ibng3) -> Inferencia YOLOv8 en paralelo por chunk    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|               FASE 3: FUSIÓN DE DATOS Y MODELADO ANALÍTICO              |
|  1. Estimación de distancia física (m) mediante geometría de Bounding Box|
|  2. Alineación temporal (pd.merge_asof) Video (ms) <-> Telemetría FIA   |
|  3. Regresión / Gradient Boosting: Cuantificación de caída de velocidad |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|             FASE 4: INTERFAZ WEB INTERACTIVA (PIT WALL UI)              |
|  Aplicación Web en Streamlit / FastAPI desplegada en el clúster:        |
|   ├── Reproductor sincronizado con detecciones YOLO y distancia (m)     |
|   ├── Gráficas interactivas de telemetría y mapa 2D del circuito        |
|   └── Monitor de métricas HPC (Speedup, Eficiencia y uso de GPUs)       |
+-------------------------------------------------------------------------+

ADAPTACIÓN A LA TOPOLOGÍA DEL CLÚSTER CIC (ibh1):
- Gestión de Recursos (SLURM): El nodo de acceso ibh1 (2 vCPUs, 3.6 GB RAM, 
  sin GPU) se utiliza exclusivamente para gestión de colas y alojamiento de la 
  interfaz web. Toda la carga de redes neuronales se delega mediante directivas 
  #SBATCH --partition=gpu --gres=gpu:4 a los nodos ibng2 e ibng3 (28 cores, 
  128 GB RAM y 4 GPUs cada uno).
- Manejo de Almacenamiento Local vs. Distribuido: Dado que los nodos de cómputo 
  operan sin sistema de archivos de red compartido (NFS) sobre /home, el 
  pipeline implementa transferencia explícita de fragmentos de video y pesos 
  del modelo hacia el almacenamiento local de alta velocidad (/tmp) de cada 
  nodo al iniciar el job, recolectando únicamente los archivos ligeros de 
  resultados (.parquet) al finalizar.

--------------------------------------------------------------------------------
3. FLUJO TÉCNICO DETALLADO (PASO A PASO)
--------------------------------------------------------------------------------

PASO 1: ADQUISICIÓN DE TELEMETRÍA OFICIAL Y VIDEO ONBOARD
- Caso de Estudio Piloto: Gran Premio de Gran Bretaña 2026 (Silverstone) — 
  Charles Leclerc (#16, Scuderia Ferrari). Este circuito cuenta con curvas 
  enlazadas de alta velocidad (Maggotts–Becketts–Chapel) donde la pérdida de 
  carga aerodinámica por turbulencia es crítica.
- Extracción Telemétrica: Mediante FastF1 (v3.9.2), se extraen los canales 
  car_data y pos_data (33,508 muestras de alta frecuencia) unificados por marca 
  de tiempo (pd.merge_asof), conservando velocidad (Speed), apertura de 
  acelerador (Throttle), presión de freno (Brake), marcha (nGear), activación 
  de DRS y coordenadas espaciales (X, Y, Z).

PASO 2: DETECCIÓN DE MONOPLAZAS Y ESTIMACIÓN DE DISTANCIA (YOLOv8)
- Entrenamiento / Fine-Tuning: Se entrena un modelo YOLOv8 especializado en 
  perspectiva onboard (T-Cam) con clases específicas: f1_car (monoplaza rival), 
  rear_wing_drs_open (alerón trasero abierto) y kerb (uso de pianos/límites de 
  pista).
- Cálculo Geométrico de Distancia: A partir de las coordenadas de la bounding 
  box detectada (x1, y1, x2, y2) y conociendo el ancho reglamentario estándar 
  de un monoplaza de F1 (W_real = 2.0 m en 2026), se estima la distancia 
  longitudinal d al auto precedente mediante proyección perspectiva calibrada:
  
      d ≈ (f_x * W_real) / w_px
  
  donde f_x es la distancia focal efectiva de la cámara T-Cam y 
  w_px = x2 - x1 es el ancho en píxeles de la detección.

PASO 3: PARALELIZACIÓN EN CLÚSTER (DATA & VIDEO PARALLELISM)
Procesar una carrera completa (90 minutos a 50/60 FPS ≈ 324,000 fotogramas) de 
manera secuencial toma horas. En el clúster aplicamos paralelismo de datos a 
nivel de flujo de video:
1) El video fuente se particiona en N segmentos independientes (sin 
   recodificación usando ffmpeg -c copy para evitar cuello de botella en CPU).
2) Un Job Array de SLURM asigna cada segmento a una GPU distinta distribuida 
   entre ibng2 e ibng3.
3) Cada proceso ejecuta inferencia por lotes (batch inference) en GPU y 
   exporta un archivo .parquet con el timestamp del fotograma, la confianza de 
   detección y la distancia estimada en metros.

PASO 4: FUSIÓN MULTIMODAL Y CUANTIFICACIÓN DEL AIRE SUCIO
1) Sincronización Temporal: Se calibra el desfase inicial (t_0) haciendo 
   coincidir el apagado de luces de salida (o el cruce por la línea de meta) en 
   el video con el aumento inicial de Speed > 0 / coordenadas de meta en FastF1.
2) Análisis Comparativo (Aire Limpio vs. Aire Sucio): Se segmentan las vueltas 
   del piloto en dos regímenes:
   - Aire Limpio (Clean Air): Distancia al rival > 50 m (sin detecciones 
     cercanas en curva).
   - Aire Sucio (Dirty Air): Distancia al rival entre 5 m y 35 m.
3) Modelado: Mediante modelos de regresión y Gradient Boosting, se calcula la 
   pérdida neta de velocidad mínima en curva (ΔV_min) y la pérdida de tiempo 
   por sector atribuible exclusivamente a la estela turbulenta del auto 
   precedente.

PASO 5: DASHBOARD WEB INTERACTIVO (FERRARI PIT WALL UI)
Para evitar una presentación estática en consola, el proyecto incluye una 
aplicación web local (desarrollada en Streamlit y accesible vía túnel SSH desde 
ibh1) dividida en 3 módulos visuales:
1) Visor Sincronizado Onboard + Telemetría: Muestra el video con las cajas de 
   YOLOv8, la distancia en metros en tiempo real y un indicador de alerta de 
   Dirty Air.
2) Mapa 2D del Circuito y Trazas: Dibuja el trazado de Silverstone usando las 
   coordenadas (X, Y) de FastF1, coloreando los sectores donde el auto perdió 
   carga aerodinámica frente a sus vueltas en aire limpio.
3) Monitor de Rendimiento HPC: Presenta las curvas experimentales de Tiempo de 
   Ejecución, Speedup (S_p = T_1 / T_p) y Eficiencia Paralela (E_p = S_p / p) 
   al escalar de 1 GPU hasta las 8 GPUs del clúster.

--------------------------------------------------------------------------------
4. ESTRUCTURA DEL REPOSITORIO
--------------------------------------------------------------------------------
f1-hpc-dirty-air/
├── app/
│   └── dashboard.py              # Interfaz Web Interactiva (Streamlit Pit Wall)
├── data/
│   ├── raw_video/                # Videos onboard fuente (.mp4)
│   ├── telemetry/                # Datasets procesados en formato Parquet
│   └── frames/                   # Dataset etiquetado para entrenamiento YOLO
├── scripts/
│   ├── 01_get_telemetry.py       # Descarga y fusión de car_data + pos_data (FastF1)
│   ├── 02_extract_frames.py      # Muestreo de fotogramas para dataset YOLO
│   ├── 03_split_video.py         # Particionado rápido de video para nodos HPC
│   ├── 04_yolo_inference.py      # Inferencia GPU y cálculo de distancia en metros
│   └── 05_merge_and_analyze.py   # Sincronización multimodal y cálculo de Dirty Air
├── slurm/
│   ├── train_yolo_ddp.slurm      # Entrenamiento distribuido Multi-GPU en ibng2/ibng3
│   └── infer_parallel.slurm      # Job Array paralelo para procesamiento de carrera
├── requirements.txt
└── README.md

--------------------------------------------------------------------------------
5. MÉTRICAS DE EVALUACIÓN (RÚBRICA ACADÉMICA HPC)
--------------------------------------------------------------------------------
Para validar el impacto del Cómputo de Alto Desempeño frente a una ejecución 
tradicional en estación de trabajo, el proyecto reporta:
- Throughput de Inferencia (FPS): Fotogramas por segundo procesados de forma 
  agregada en el clúster.
- Speedup (S_p): Aceleración obtenida al escalar de 1 GPU a 2, 4 y 8 GPUs 
  entre los nodos ibng2 e ibng3.
- Precisión del Modelo (mAP@50): Exactitud de YOLOv8 detectando monoplazas a 
  distintas distancias y condiciones de iluminación.
- Correlación Física (R^2): Relación estadística entre la distancia visual 
  estimada y la caída de velocidad en curva rápida.
================================================================================