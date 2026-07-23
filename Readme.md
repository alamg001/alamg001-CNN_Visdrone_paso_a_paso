# CNN_Visdrone_paso_a_paso

## Algoritmo

```latex
\documentclass[20pt]{beamer}

% ============================================================
% LATEX_UNI: presentación Beamer autocontenida, A4 horizontal,
% compatible con pdfLaTeX, MiKTeX y TeX Live.
% ============================================================

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-tabla]{babel}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage[most]{tcolorbox}
\usepackage{ragged2e}
\usepackage{hyperref}
\usetikzlibrary{positioning,fit,calc,arrows.meta,shapes.geometric}

\usetheme{Madrid}
\usecolortheme{beaver}

\setbeamertemplate{frametitle continuation}{}

% A4 horizontal real: 297 x 210 mm
\setlength{\paperwidth}{297mm}
\setlength{\paperheight}{210mm}
\geometry{paperwidth=297mm,paperheight=210mm}
\setbeamersize{text margin left=8mm,text margin right=8mm}

% Paleta institucional DIMRed
\definecolor{DIMRed}{HTML}{BA574A}
\definecolor{DIMRedDark}{HTML}{7E332B}
\definecolor{DIMRedLight}{HTML}{F4E1DE}
\definecolor{DIMGray}{HTML}{3B3B3B}

\setbeamercolor{structure}{fg=DIMRed}
\setbeamercolor{frametitle}{fg=white,bg=DIMRed}
\setbeamercolor{title}{fg=white,bg=DIMRed}
\setbeamercolor{block title}{fg=white,bg=DIMRed}
\setbeamercolor{block body}{bg=DIMRedLight!35}
\setbeamercolor{normal text}{fg=DIMGray}
\setbeamercolor{author in head/foot}{fg=white,bg=DIMRedDark}
\setbeamercolor{date in head/foot}{fg=white,bg=DIMRed}

\hypersetup{
  pdftitle={Análisis Vectorial y Matricial de una CNN para VisDrone},
  pdfauthor={Antonio Lam G.},
  pdfsubject={Redes neuronales convolucionales, PyTorch y álgebra lineal},
  pdfkeywords={CNN, PyTorch, VisDrone, álgebra lineal, aprendizaje profundo},
  colorlinks=true,
  linkcolor=DIMRedDark,
  urlcolor=DIMRedDark,
  citecolor=DIMRedDark
}

% Pie institucional
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{%
  \leavevmode%
  \hbox{%
    \begin{beamercolorbox}[wd=.50\paperwidth,ht=2.6ex,dp=1.2ex,leftskip=1.2em]{author in head/foot}
      \scriptsize Doctorando: Antonio Lam G.
    \end{beamercolorbox}%
    \begin{beamercolorbox}[wd=.50\paperwidth,ht=2.6ex,dp=1.2ex,rightskip=1.2em plus1fil]{date in head/foot}
      \hfill\scriptsize \today\quad\insertframenumber/\inserttotalframenumber
    \end{beamercolorbox}%
  }%
}

% Cajas académicas canónicas
\tcbset{
  enhanced,
  sharp corners,
  boxrule=0.7pt,
  colframe=DIMRed,
  colback=DIMRedLight!35,
  coltitle=white,
  colbacktitle=DIMRed,
  fonttitle=\bfseries,
  before skip=0.6em,
  after skip=0.6em
}
\newtcolorbox{defbox}[1]{breakable,title={Definición: #1}}
\newtcolorbox{theobox}[1]{breakable,title={Teorema: #1}}
\newtcolorbox{lemmabox}[1]{breakable,title={Lema: #1}}
\newtcolorbox{corbox}[1]{breakable,title={Corolario: #1}}
\newtcolorbox{resbox}[1][]{breakable,colback=white,#1}
\newtcolorbox{warnbox}[1]{breakable,title={Advertencia metodológica: #1},colframe=DIMRedDark,colback=DIMRedLight!55}
\newtcolorbox{propbox}[1]{breakable,title={Propuesta: #1},colframe=DIMRedDark,colback=white}
\newtcolorbox{chapterbox}[1]{breakable,title={#1},colframe=DIMRedDark,colback=white}

\setlength{\emergencystretch}{3em}
\setlength{\abovedisplayskip}{0.45em}
\setlength{\belowdisplayskip}{0.45em}
\setlength{\abovedisplayshortskip}{0.3em}
\setlength{\belowdisplayshortskip}{0.3em}

\title[CNN y álgebra lineal en VisDrone]{Análisis Vectorial y Matricial de una Red Neuronal Convolucional para Clasificación de Objetos en VisDrone}
\subtitle{Estructura matemática de una CNN implementada con PyTorch}
\author{Antonio Lam G.}
\institute{Informe técnico doctoral en estadística y aprendizaje profundo}
\date{\today}

\begin{document}

\begin{frame}
  \titlepage
\end{frame}

\begin{frame}[allowframebreaks]{Resumen ejecutivo} %🔴🔴🔴🔴🔴🔴
\justifying
En el presente informe se presenta un análisis vectorial y matricial meticuloso del notebook \texttt{cnn\_visdrone\_paso\_paso.ipynb}. Se descompone cada etapa del flujo de trabajo de una Red Neuronal Convolucional (CNN) en sus componentes fundamentales de álgebra lineal: espacios vectoriales, transformaciones lineales, productos punto, productos matriciales, y operaciones de reducción de dimensionalidad. Se proporcionan representaciones gráficas 3D para ilustrar la geometría de los espacios de características y la dinámica del gradiente descendente. El análisis se realiza desde la perspectiva de un Doctor en Estadística Matemática, enfatizando la estructura matemática subyacente que permite a la CNN aprender representaciones jerárquicas de los datos visuales, y cómo esta comprensión vectorial es esencial para la transición hacia modelos de detección de objetos como YOLO.
\end{frame}

\begin{frame}[allowframebreaks, t]{Contenido} %⚽⚽⚽⚽⚽⚽⚽⚽⚽⚽⚽⚽⚽⚽⚽⚽⚽⚽⚽
	\tableofcontents
\end{frame}


\section{Introducción: El Espacio Vectorial de las Imágenes}

%📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌
\begin{frame}{Introducción: El Espacio Vectorial de las Imágenes}
\justifying
\begin{chapterbox}{Panorama del capítulo}
El notebook \texttt{cnn\_visdrone\_paso\_paso.ipynb} constituye una exploración fundamental del aprendizaje profundo aplicado a la visión por computadora. Su objetivo principal es construir una Red Neuronal Convolucional (CNN) desde cero para clasificar objetos del dataset VisDrone, estableciendo un puente conceptual hacia sistemas de detección de objetos como YOLO. Para comprender verdaderamente el funcionamiento de esta red, es imperativo descomponer cada operación en sus componentes matemáticos subyacentes: vectores, matrices, transformaciones lineales y espacios vectoriales.
\end{chapterbox}
\end{frame}
%📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌

%\subsection{Representación Vectorial de una Imagen}

\begin{frame}[allowframebreaks]{Representación Vectorial de una Imagen}
\justifying
Una imagen digital en color se representa naturalmente como un tensor tridimensional. Sin embargo, para el análisis matemático, es conveniente vectorizarla. Una imagen de tamaño \(H \times W\) con \(C\) canales (RGB) puede ser vista como un vector en un espacio de alta dimensión:
\begin{equation}
    \mathbf{x} \in \mathbb{R}^{C \times H \times W} \quad \text{o vectorizado como} \quad \mathbf{x} \in \mathbb{R}^{C \cdot H \cdot W}
\end{equation}
Para una imagen de \(96 \times 96\) píxeles con 3 canales, tenemos \(\mathbf{x} \in \mathbb{R}^{27648}\). Cada píxel \(x_{c,i,j}\) representa una coordenada en este espacio de 27\,648 dimensiones.
\end{frame}

%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
\begin{frame}[allowframebreaks]{Representación Vectorial de una Imagen}
\begin{figure}
   \centering
    \includegraphics[width=0.9 \textwidth]{Figura_1_1.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%\begin{resbox}
%\textbf{Figura 1.1: Espacio Vectorial de una Imagen.} Una imagen $96\times 96$ en escala de grises se representa como un vector en \(\mathbb{R}^27648\). Cada píxel es una coordenada. La intensidad del color determina el valor de la coordenada. En el caso de %imágenes RGB, el espacio se vuelve tridimensional por canal.
%\end{resbox}
\end{frame}
%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅


\subsection{Geometría del Espacio de Características}
\begin{frame}[allowframebreaks]{Geometría del Espacio de Características}
\justifying
El objetivo de una CNN es aprender una transformación que mapee el espacio de entrada de alta dimensión a un espacio de características donde las clases sean linealmente separables o fácilmente distinguibles. Cada capa de la red aplica una transformación no lineal que deforma y reconfigura este espacio.

\begin{equation}
    f: \mathbb{R}^{C \times H \times W} \to \mathbb{R}^{d}
\end{equation}

Donde \(d\) es la dimensionalidad del espacio de características final (en nuestro caso, \(d=10\) para las 10 clases). Esta función \(f\) se compone de una secuencia de transformaciones lineales (convoluciones, capas lineales) y no lineales (ReLU, MaxPooling).
\end{frame}


%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{ Mapeo del Espacio de Entrada al Espacio de Características}
\begin{figure}
   \centering
    \includegraphics[width=0.9 \textwidth]{Figura_1_2.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%\begin{resbox}
%\textbf{Figura 1.1: Espacio Vectorial de una Imagen.} Una imagen $96\times 96$ en escala de grises se representa como un vector en \(\mathbb{R}^27648\). Cada píxel es una coordenada. La intensidad del color determina el valor de la coordenada. En el caso de %imágenes RGB, el espacio se vuelve tridimensional por canal.
%\end{resbox}
\end{frame}
%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2

\begin{frame}[allowframebreaks]{ Mapeo del Espacio de Entrada al Espacio de Características}
\begin{resbox}
\textbf{Figura 1.2: Mapeo del Espacio de Entrada al Espacio de Características.} \ \\ 
La CNN aprende una función no lineal que transforma el espacio de entrada (imágenes) en un espacio de características de menor dimensionalidad donde las clases están separadas. La superficie de decisión es aprendida durante el entrenamiento.
\end{resbox}
\end{frame}



\subsection{Ventaja de PyTorch en el Espacio Vectorial}
\begin{frame}[allowframebreaks]{Ventaja de PyTorch en el Espacio Vectorial}
\justifying
PyTorch se destaca sobre otras librerías como NumPy por su capacidad para operar eficientemente en estos espacios vectoriales de alta dimensión mediante:

\begin{enumerate}
    \item \textbf{Cómputo Diferenciable (Autograd):} PyTorch construye un grafo computacional dinámico que permite calcular gradientes de forma automática.
    \item \textbf{Aceleración por GPU:} Las operaciones matriciales y tensoriales se ejecutan en paralelo en GPUs.
\end{enumerate}

La representación de datos como \texttt{Tensor}s en PyTorch es análoga a trabajar con vectores y matrices en un sentido matemático, pero con la ventaja de que las operaciones son altamente optimizadas y diferenciables.
\end{frame}


%%🏈


%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{Ventaja de PyTorch en el Espacio Vectorial}
\begin{figure}
   \centering
    \includegraphics[width=0.90 \textwidth]{pytorch.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%\begin{resbox}
%\textbf{Figura 1.1: Espacio Vectorial de una Imagen.} Una imagen $96\times 96$ en escala de grises se representa como un vector en \(\mathbb{R}^27648\). Cada píxel es una coordenada. La intensidad del color determina el valor de la coordenada. En el caso de %imágenes RGB, el espacio se vuelve tridimensional por canal.
%\end{resbox}
\end{frame}
%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2

\begin{frame}[allowframebreaks]{Ventaja de PyTorch en el Espacio Vectorial}
\begin{resbox}
\textbf{Figura 1.3: Ventaja de PyTorch en el Espacio Vectorial.}
\ \\
Para VisDrone, PyTorch actúa como una plataforma de álgebra tensorial diferenciable. Permite representar cada imagen como un tensor de alta dimensión, transformar ese tensor mediante convoluciones, calcular automáticamente gradientes con Autograd y aprovechar el paralelismo de la GPU. La combinación de estas propiedades hace viable el entrenamiento de modelos profundos capaces de extraer patrones visuales complejos a partir de escenas aéreas densas y heterogéneas.
\end{resbox}
\end{frame}


%%🏈

\section{Preparación Vectorial de los Datos}
\begin{frame}{Preparación Vectorial de los Datos}
\justifying
\begin{chapterbox}{Panorama del capítulo}
La preparación de los datos es el primer paso donde se transforma el espacio de las imágenes anotadas en un espacio vectorial estructurado para el aprendizaje automático.
\end{chapterbox}
\end{frame}

\subsection{De la Caja Delimitadora al Vector de Características}
\begin{frame}[allowframebreaks]{De la Caja Delimitadora al Vector de Características}
\justifying
El proceso implementado en la clase \texttt{VisDroneCropsDataset} realiza las siguientes operaciones vectoriales:

\medskip\textcolor{DIMRedDark}{\large\bfseries Recorte (Crop)}\par\smallskip
Para cada anotación con coordenadas \((x_1, y_1, x_2, y_2)\), se extrae una submatriz de la imagen original:
\begin{equation}
    \mathbf{X}_{\text{recorte}} = \mathbf{X}[y_1:y_2, x_1:x_2, :] \in \mathbb{R}^{3 \times h \times w}
\end{equation}

\medskip\textcolor{DIMRedDark}{\large\bfseries Redimensionamiento (Resize)}\par\smallskip
El recorte se redimensiona a \(S \times S\) (\(S=96\)). Este es un mapeo no lineal del espacio de recortes de tamaño variable al espacio fijo:
\begin{equation}
    \text{Resize}: \mathbb{R}^{3 \times h \times w} \to \mathbb{R}^{3 \times S \times S}
\end{equation}

\medskip\textcolor{DIMRedDark}{\large\bfseries Normalización y Conversión a Tensor}\par\smallskip
La normalización escala el espacio vectorial al rango \([0,1]\):
\begin{equation}
    \mathbf{X}_{\text{norm}} = \frac{\mathbf{X}_{\text{recorte}}}{255.0} \in [0,1]^{3 \times S \times S}
\end{equation}

Finalmente, la permutación de dimensiones reorganiza el espacio:
\begin{equation}
    \text{Permute}: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{C \times H \times W}
\end{equation}

\end{frame}


%%🏈


%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{De la Caja Delimitadora al Vector de Características}
\begin{figure}
   \centering
    \includegraphics[width=1 \textwidth]{Figura_2_1.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%\begin{resbox}
%\textbf{Figura 1.1: Espacio Vectorial de una Imagen.} Una imagen $96\times 96$ en escala de grises se representa como un vector en \(\mathbb{R}^27648\). Cada píxel es una coordenada. La intensidad del color determina el valor de la coordenada. En el caso de %imágenes RGB, el espacio se vuelve tridimensional por canal.
%\end{resbox}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2


\begin{resbox}
\textbf{Figura 2.1: Transformaciones del Espacio Vectorial de Imágenes.} Una imagen se recorta, redimensiona, normaliza y se convierte en un tensor de PyTorch. Cada operación es una transformación del espacio vectorial original al espacio de entrada de la red.
\end{resbox}

\end{frame}


%🏈🏈

\subsection{Estructura del Lote (Batch)}
\begin{frame}[allowframebreaks]{Estructura del Lote (Batch)}
\justifying
El \texttt{DataLoader} agrupa múltiples muestras en un lote, creando un tensor de 4 dimensiones:
\begin{equation}
    \mathbf{X}_{\text{batch}} \in \mathbb{R}^{B \times C \times H \times W}
\end{equation}
Donde \(B\) es el tamaño del lote. Esto permite la vectorización de las operaciones, procesando \(B\) vectores de entrada simultáneamente.

\end{frame}
%%🏈


%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{Estructura del Lote (Batch)}
\begin{figure}
   \centering
    \includegraphics[width=0.9 \textwidth]{Figura_2_2.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%\begin{resbox}
%\textbf{Figura 1.1: Espacio Vectorial de una Imagen.} Una imagen $96\times 96$ en escala de grises se representa como un vector en \(\mathbb{R}^27648\). Cada píxel es una coordenada. La intensidad del color determina el valor de la coordenada. En el caso de %imágenes RGB, el espacio se vuelve tridimensional por canal.
%\end{resbox}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2



\begin{resbox}
\textbf{Figura 2.2: Espacio de Lotes (Batch Space).} Un lote de imágenes se representa como un tensor 4D. La primera dimensión es el índice de la muestra en el lote. Las operaciones se vectorizan a lo largo de esta dimensión.
\end{resbox}
\end{frame}

\section{El Espacio Vectorial de PyTorch}
\begin{frame}{El Espacio Vectorial de PyTorch}
\justifying
\begin{chapterbox}{Panorama del capítulo}
PyTorch implementa un espacio vectorial tensorial que generaliza los conceptos de vectores y matrices a dimensiones superiores. Este capítulo explora la naturaleza vectorial de las operaciones fundamentales en PyTorch.
\end{chapterbox}
\end{frame}

\subsection{Tensores como Vectores y Matrices}
\begin{frame}[allowframebreaks]{Tensores como Vectores y Matrices}
\justifying
Un tensor de PyTorch puede ser visto como un vector en un espacio de alta dimensión o como una matriz cuando se considera su forma (shape).

\begin{itemize}
    \item \textbf{Tensor 1D:} Vector \(\mathbf{v} \in \mathbb{R}^n\)
    \item \textbf{Tensor 2D:} Matriz \(\mathbf{M} \in \mathbb{R}^{m \times n}\)
    \item \textbf{Tensor 3D:} Arreglo de matrices o un cubo de datos \(\mathbf{T} \in \mathbb{R}^{c \times h \times w}\)
\end{itemize}
\end{frame}

\subsection{Operaciones Vectoriales en PyTorch}
\begin{frame}[allowframebreaks]{Operaciones Vectoriales en PyTorch}
\justifying
Las operaciones fundamentales en PyTorch son extensiones de las operaciones vectoriales y matriciales clásicas.

\medskip\textcolor{DIMRedDark}{\large\bfseries Producto Punto (Dot Product)}\par\smallskip
El producto punto entre dos vectores se realiza con la función \texttt{torch.dot} o mediante el operador \texttt{@} para matrices:
\begin{equation}
    \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \mathbf{a}^T \mathbf{b}
\end{equation}

\medskip\textcolor{DIMRedDark}{\large\bfseries Producto Matricial (Matrix Multiplication)}\par\smallskip
El producto de matrices se realiza con \texttt{torch.mm} o \texttt{@}:
\begin{equation}
    \mathbf{C} = \mathbf{A} \mathbf{B} \quad \text{donde} \quad C_{i,j} = \sum_{k=1}^{m} A_{i,k} B_{k,j}
\end{equation}

\medskip\textcolor{DIMRedDark}{\large\bfseries Broadcasting}\par\smallskip
El broadcasting es una extensión de las operaciones vectoriales que permite operar con tensores de diferentes formas, replicando automáticamente los datos a lo largo de dimensiones faltantes.

\end{frame}

%🏈


%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{Convolución como producto matricial con matriz de Toeplitz}
\begin{figure}
   \centering
    \includegraphics[width=1 \textwidth]{matriz_toeplitz_v4.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2


%🏈

\begin{resbox}
\textbf{Figura 3.1: Convolución como producto matricial con matriz de Toeplitz.} 

Cada fila de la matriz de Toeplitz $T(K)$ representa el kernel aplicado en una posición distinta de la imagen. Al multiplicar $T(K)$ por la imagen vectorizada $\operatorname{vec}(X)$, cada fila produce un valor del mapa de características. Los nueve resultados obtenidos se reorganizan para formar la matriz $Y\in\mathbb{R}^{3\times3}$.

\end{resbox}
\end{frame}

\subsection{Espacio Vectorial de Parámetros}
\begin{frame}[allowframebreaks]{Espacio Vectorial de Parámetros}
\justifying
Los parámetros de la red, \(\boldsymbol{\theta}\), viven en un espacio de alta dimensión \(\mathbb{R}^P\), donde \(P\) es el número total de parámetros entrenables.

\begin{equation}
    \boldsymbol{\theta} \in \mathbb{R}^P
\end{equation}

Cada punto en este espacio representa una configuración completa de la CNN. El entrenamiento es un proceso de búsqueda en este espacio de alta dimensión para encontrar el punto que minimiza la función de pérdida.
\end{frame}


%🏈
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{Espacio de Par\'ametros de la CNN}
\begin{figure}
   \centering
    \includegraphics[width=0.9 \textwidth]{figura_3_2.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2
\end{frame}

%🏈


\begin{frame}[allowframebreaks]{Espacio de Par\'ametros de la CNN}
\begin{resbox}
\textbf{Figura 3.2: Espacio de Parámetros de la CNN.} Cada punto en este espacio de alta dimensión representa una instancia de la CNN con pesos y sesgos específicos. El entrenamiento busca el punto que minimiza la función de pérdida.
\end{resbox}
\end{frame}



\subsection{Computación Vectorizada}
\begin{frame}[allowframebreaks]{Computación Vectorizada}
\justifying
La vectorización es la clave de la eficiencia en PyTorch. En lugar de procesar una muestra a la vez, se procesan lotes completos:
\begin{equation}
    \mathbf{Y}_{\text{batch}} = f(\mathbf{X}_{\text{batch}})
\end{equation}
Donde \(f\) se aplica a cada elemento del lote en paralelo. Esto es posible gracias a la naturaleza vectorial de las operaciones.
\end{frame}


%🏈
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{Computación Vectorizada}
\begin{figure}
   \centering
    \includegraphics[width= 0.9 \textwidth]{computacion_vectorizada.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2
\end{frame}

%🏈

% --- Aquí forzamos un salto manual en el índice ---
\addtocontents{toc}{\protect\framebreak}  %.😎😎😎😎😎😎😎😎😎😎😎

\section{Arquitectura Matricial de la CNN}
\begin{frame}{Arquitectura Matricial de la CNN}
\justifying
\begin{chapterbox}{Panorama del capítulo}
La CNN implementada en el notebook es una secuencia de capas convolucionales y de agrupamiento. Cada capa realiza transformaciones vectoriales que mapean el espacio de entrada a un nuevo espacio de características.
\end{chapterbox}
\end{frame}


%🏈%🏈%🏈%🏈%🏈%🏈%🏈%🏈%🏈%🏈%🏈%🏈%🏈%🏈%🏈


\subsection{La Capa Convolucional como Transformación Lineal}
\begin{frame}[allowframebreaks]{La Capa Convolucional como Transformación Lineal}
\justifying
La capa convolucional \texttt{nn.Conv2d} es el corazón de la CNN. Aunque no es una transformación lineal en el sentido estricto debido a la no linealidad de la activación (ReLU), su componente fundamental (la convolución) es una operación lineal.

\medskip\textcolor{DIMRedDark}{\large\bfseries Convolución como Producto Matricial}\par\smallskip

La operación de convolución puede ser expresada como un producto matricial. Sea \(\mathbf{X}\) la imagen de entrada y \(\mathbf{W}\) un filtro (kernel). La convolución \(\mathbf{Y} = \mathbf{W} * \mathbf{X}\) puede ser representada como:

\begin{equation}
    \mathbf{Y} = \mathbf{W}_{\text{toeplitz}} \cdot \mathbf{X}_{\text{vec}}
\end{equation}

Donde \(\mathbf{W}_{\text{toeplitz}}\) es una matriz de Toeplitz (o bloque-circulante) que representa el filtro, y \(\mathbf{X}_{\text{vec}}\) es el vector de entrada.

\end{frame}
%🏈

%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{Convolucional como Transformación Lineal}
\begin{figure}
   \centering
    \includegraphics[width= 0.9 \textwidth]{convolucion.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2
\end{frame}


\begin{frame}[allowframebreaks]{Convolucional como Transformación Lineal}
%🏈
\begin{resbox}
\textbf{Figura 4.1: Convolución como Producto Matricial.} La convolución 2D se puede descomponer en un producto matriz-vector. La matriz de Toeplitz contiene los pesos del filtro, y el vector contiene los píxeles de la imagen. Este es un ejemplo de cómo una operación de deslizamiento se convierte en álgebra lineal.
\end{resbox}
\end{frame}

%🏈%🏈%🏈%🏈%🏈%🏈

\begin{frame}[allowframebreaks]{Convolucional como Transformación Lineal}
\medskip\textcolor{DIMRedDark}{\large\bfseries Producto Punto en la Convolución}\par\smallskip

Para cada posición \((i,j)\), la convolución calcula el producto punto entre el filtro \(\mathbf{W}\) (vectorizado) y la ventana de la imagen \(\mathbf{X}_{i,j}\) (vectorizada):

\begin{equation}
    y_{i,j} = \langle \text{vec}(\mathbf{W}), \text{vec}(\mathbf{X}_{i,j}) \rangle + b
\end{equation}

Este es el producto punto clásico entre dos vectores en \(\mathbb{R}^{K^2}\).

\medskip\textcolor{DIMRedDark}{\large\bfseries Representación Matricial Completa de la Capa}\par\smallskip

Para una capa convolucional con \(C_{\text{in}}\) canales de entrada y \(C_{\text{out}}\) filtros:

\begin{equation}
    \mathbf{Y}^{(k)} = \sigma \left( \sum_{c=1}^{C_{\text{in}}} \mathbf{W}^{(k,c)} * \mathbf{X}^{(c)} + b^{(k)} \right)
\end{equation}

Donde \(\mathbf{W}^{(k,c)}\) es el filtro \(k\) para el canal de entrada \(c\). En forma matricial:

\begin{equation}
    \mathbf{Y}_{\text{vec}} = \sigma \left( \mathbf{W}_{\text{conv}} \mathbf{X}_{\text{vec}} + \mathbf{b}_{\text{vec}} \right)
\end{equation}

Donde \(\mathbf{W}_{\text{conv}}\) es una matriz que contiene todos los filtros en una estructura de bloques.

\end{frame}

%🏈🏈🏈🏈🏈🏈


%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{Convolucional como Transformación Lineal}
\begin{figure}
   \centering
    \includegraphics[width= 0.9 \textwidth]{transformacion.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2
\end{frame}


%🏈🏈🏈🏈🏈🏈

\begin{frame}[allowframebreaks]{Convolucional como Transformación Lineal}
\begin{resbox}
\textbf{Figura 4.2: Transformación del Espacio de Características por una Capa Convolucional.} La capa convolucional mapea el espacio de entrada (izquierda) a un nuevo espacio de características (derecha). Cada filtro detecta un patrón específico, y la combinación de filtros permite una representación más rica.
\end{resbox}
\end{frame}

\subsection{Capa de Agrupamiento Máximo (MaxPooling)}
\begin{frame}[allowframebreaks]{Capa de Agrupamiento Máximo (MaxPooling)}
\justifying
El \texttt{MaxPool2d} reduce la dimensionalidad espacial. Para una ventana de \(2 \times 2\), la operación es:

\begin{equation}
    y_{i,j} = \max\{ x_{2i,2j}, x_{2i+1,2j}, x_{2i,2j+1}, x_{2i+1,2j+1} \}
\end{equation}

Esta es una operación de reducción de dimensionalidad no lineal que preserva las características más prominentes.

\medskip\textcolor{DIMRedDark}{\large\bfseries Reducción de Dimensionalidad en el Espacio Vectorial}\par\smallskip

El MaxPooling reduce la dimensionalidad del espacio de características de \(\mathbb{R}^{C \times H \times W}\) a \(\mathbb{R}^{C \times H/2 \times W/2}\).

\end{frame}
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{Capa de Agrupamiento Máximo (MaxPooling)}
\begin{figure}
   \centering
    \includegraphics[width= 0.9 \textwidth]{Figura_4_3.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2
\end{frame}


%🏈🏈🏈🏈🏈🏈
\begin{frame}[allowframebreaks]{Capa de Agrupamiento Máximo (MaxPooling)}
\begin{resbox}
\textbf{Figura 4.3: Reducción de Dimensionalidad por MaxPooling.} El MaxPooling reduce la resolución espacial, disminuyendo la dimensionalidad del espacio de características y proporcionando invariancia a pequeñas traslaciones.
\end{resbox}
\end{frame}

\subsection{Secuencia de Transformaciones Vectoriales}
\begin{frame}[allowframebreaks]{Secuencia de Transformaciones Vectoriales}
\justifying
La CNN completa es una composición de transformaciones:

\begin{align}
  \mathbf{h}_0 &= \sigma_0\!\left(\mathbf{W}_1 * \mathbf{x} + \mathbf{b}_1\right),\\
  \mathbf{h}_1 &= \sigma_1\!\left(\operatorname{Pool}_1(\mathbf{h}_0)\right),\\
  \mathbf{h}_2 &= \sigma_2\!\left(\operatorname{Pool}_2(\mathbf{h}_1)\right),\\
  \mathbf{h}_3 &= \sigma_3\!\left(\operatorname{Pool}_3(\mathbf{h}_2)\right),\\
  f(\mathbf{x}) &= \mathbf{W}_{fc}\mathbf{h}_3 + \mathbf{b}_{fc}.
\end{align}

Cada \(\sigma_k\) es la función ReLU, y cada \(\text{Pool}_k\) es MaxPooling.


%🏈
\end{frame}
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{Secuencia de Transformaciones Vectoriales}
\begin{figure}
   \centering
    \includegraphics[width= 0.9 \textwidth]{Secuencia_Tranf_Lineales_CNN.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2
\end{frame}

%🏈

\begin{frame}[allowframebreaks]{Secuencia de Transformaciones Vectoriales}
\begin{resbox}
\textbf{Figura 4.4: Espacio Vectorial a través de las Capas de la CNN.} Visualización de cómo el espacio de características se transforma a través de las capas convolucionales y de pooling. La dimensionalidad espacial se reduce mientras que la profundidad (número de canales) aumenta.
\end{resbox}
\end{frame}

\subsection{Extracción de Características y el Espacio de Características}
\begin{frame}[allowframebreaks]{Extracción de Características y el Espacio de Características}
\justifying
El poder de una CNN reside en su capacidad para aprender características jerárquicas. Las capas tempranas aprenden características de bajo nivel (bordes, colores), mientras que las capas profundas aprenden características de alto nivel (partes de objetos, objetos completos).

\begin{equation}
\begin{aligned}
  \mathbf{x}   &\xrightarrow{\text{Capa 1}} \mathbf{h}_1
    && \text{(características de bajo nivel)},\\
  \mathbf{h}_1 &\xrightarrow{\text{Capa 2}} \mathbf{h}_2
    && \text{(características de nivel medio)},\\
  \mathbf{h}_2 &\xrightarrow{\text{Capa 3}} \mathbf{h}_3
    && \text{(características de alto nivel)}.
\end{aligned}
\end{equation}


%🏈
\end{frame}
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{Extracción de Características y el Espacio de Características}
\begin{figure}
   \centering
    \includegraphics[width= 1 \textwidth]{Figura_4_5.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2
\end{frame}

%🏈

\begin{frame}[allowframebreaks]{Extracción de Características y el Espacio de Características}


\begin{resbox}
\textbf{Figura 4.5: Jerarquía de Características en el Espacio Vectorial.} Cada capa de la CNN aprende una representación más abstracta de la entrada. El espacio de características se transforma progresivamente, aprendiendo características cada vez más complejas.
\end{resbox}
\end{frame}

\subsection{Importancia de los Features en el Espacio Vectorial}
\begin{frame}[allowframebreaks]{Importancia de los Features en el Espacio Vectorial}
\justifying
Los \textit{features} son las coordenadas en el espacio de características aprendido. La calidad de estos \textit{features} determina el rendimiento de la red. La CNN ajusta sus parámetros para que los \textit{features} de diferentes clases estén bien separados en el espacio de características, facilitando la clasificación.


%🏈
\end{frame}
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{Importancia de los Features en el Espacio Vectorial}
\begin{figure}
   \centering
    \includegraphics[width= 0.9 \textwidth]{Figura_4_6.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2
\end{frame}

\begin{frame}[allowframebreaks]{Importancia de los Features en el Espacio Vectorial}
%🏈

\begin{resbox}
\textbf{Figura 4.6: Separabilidad de Clases en el Espacio de Características.} Una buena CNN transforma el espacio de entrada de modo que las diferentes clases estén separadas linealmente o sean fácilmente distinguibles en el espacio de características final.
\end{resbox}
\end{frame}



\section{Optimización Vectorial: Gradiente Descendente}
\begin{frame}{Optimización Vectorial: Gradiente Descendente}
\justifying
\begin{chapterbox}{Panorama del capítulo}
El entrenamiento de una CNN es un problema de optimización en el espacio de parámetros \(\mathbb{R}^P\). La función objetivo es la pérdida empírica.
\end{chapterbox}
\end{frame}

\subsection{La Función de Pérdida como Superficie en \texorpdfstring{\(\mathbb{R}^P\)}{R elevado a P}}
\begin{frame}[allowframebreaks]{La Función de Pérdida como Superficie en \texorpdfstring{\(\mathbb{R}^P\)}{R elevado a P}}
\justifying
La función de pérdida \(J(\boldsymbol{\theta})\) mapea cada punto en el espacio de parámetros a un escalar:

\begin{equation}
    J: \mathbb{R}^P \to \mathbb{R}
\end{equation}

Esta función define una superficie (o hipersuperficie) en \(\mathbb{R}^{P+1}\). El objetivo es encontrar el mínimo de esta superficie.


%🏈
\end{frame}
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{La Función de Pérdida como Superficie en \texorpdfstring{\(\mathbb{R}^P\)}{R elevado a P}}
\begin{figure}
   \centering
    \includegraphics[width= 0.9 \textwidth]{Figura_5_1.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2
\end{frame}

\begin{frame}[allowframebreaks]{La Función de Pérdida como Superficie en \texorpdfstring{\(\mathbb{R}^P\)}{R elevado a P}}
%🏈

\begin{resbox}
\textbf{Figura 5.1: Superficie de Pérdida en el Espacio de Parámetros.} La función de pérdida define una superficie en el espacio de parámetros. Los mínimos locales y globales corresponden a buenas configuraciones de la red.
\end{resbox}
\end{frame}

\subsection{El Gradiente como Vector en \texorpdfstring{\(\mathbb{R}^P\)}{R elevado a P}}
\begin{frame}[allowframebreaks]{El Gradiente como Vector en \texorpdfstring{\(\mathbb{R}^P\)}{R elevado a P}}
\justifying
El gradiente de la pérdida con respecto a los parámetros es un vector en el espacio de parámetros:

\begin{equation}
    \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \begin{bmatrix}
        \frac{\partial J}{\partial \theta_1} \\
        \frac{\partial J}{\partial \theta_2} \\
        \vdots \\
        \frac{\partial J}{\partial \theta_P}
    \end{bmatrix} \in \mathbb{R}^P
\end{equation}

Cada componente del gradiente indica la dirección y magnitud del cambio en la pérdida con respecto a ese parámetro específico.


%🏈
\end{frame}
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 1.2

\begin{frame}[allowframebreaks]{El Gradiente como Vector en \texorpdfstring{\(\mathbb{R}^P\)}{R elevado a P}}
\begin{figure}
   \centering
    \includegraphics[width= 0.9 \textwidth]{Figura_5_2.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 1.2
\end{frame}

\begin{frame}[allowframebreaks]{El Gradiente como Vector en \texorpdfstring{\(\mathbb{R}^P\)}{R elevado a P}}
%🏈

\begin{resbox}
\textbf{Figura 5.2: Gradiente como Vector en el Espacio de Parámetros.} El gradiente es un vector que apunta en la dirección de máximo crecimiento de la función de pérdida. Para minimizar, nos movemos en la dirección opuesta.
\end{resbox}
\end{frame}


\subsection{Actualización de Parámetros como Operación Vectorial}
\begin{frame}[allowframebreaks]{Actualización de Parámetros como Operación Vectorial}
\justifying
La regla de actualización del gradiente descendente es:

\begin{equation}
    \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_t)
\end{equation}

Esta es una operación vectorial en \(\mathbb{R}^P\): restar un vector (el gradiente escalado) de otro vector (los parámetros actuales).

\medskip\textcolor{DIMRedDark}{\large\bfseries Interpretación Geométrica}\par\smallskip

Cada paso del gradiente descendente mueve el punto representativo en el espacio de parámetros en la dirección de máxima pendiente negativa, descendiendo por la superficie de pérdida.



%🏈
\end{frame}
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 5.3

\begin{frame}[allowframebreaks]{Actualización de Parámetros como Operación Vectorial}
\begin{figure}
   \centering
    \includegraphics[width= 0.9 \textwidth]{Figura_5_3.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 5.3
\end{frame}

\begin{frame}[allowframebreaks]{Actualización de Parámetros como Operación Vectorial}
%🏈

\begin{resbox}
\textbf{Figura 5.3: Descenso del Gradiente en el Espacio de Parámetros.} Visualización del proceso de descenso del gradiente. Los puntos representan las iteraciones, moviéndose hacia un mínimo de la superficie de pérdida. La trayectoria muestra la evolución de los parámetros durante el entrenamiento.
\end{resbox}
\end{frame}

\subsection{Adam: Optimización con Momentos}
\begin{frame}[allowframebreaks]{Adam: Optimización con Momentos}
\justifying
Adam (Adaptive Moment Estimation) es una extensión del gradiente descendente que utiliza momentos de primer y segundo orden para adaptar la tasa de aprendizaje para cada parámetro.

\begin{align}
    \mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_t) \\
    \mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_t))^2 \\
    \hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t} \\
    \hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1 - \beta_2^t} \\
    \boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\end{align}

Cada operación es vectorial en \(\mathbb{R}^P\).


%🏈
\end{frame}
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 5.4

\begin{frame}[allowframebreaks]{Figura 5.4: Comparación de Trayectorias de Optimización.}
\begin{figure}
   \centering
    \includegraphics[width= 0.9 \textwidth]{Figura_5_4.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 5.4
\end{frame}

\begin{frame}[allowframebreaks]{Actualización de Parámetros como Operación Vectorial}
%🏈

\begin{resbox}
\textbf{Figura 5.4: Comparación de Trayectorias de Optimización.} Diferentes algoritmos de optimización (SGD, Adam) toman diferentes trayectorias en el espacio de parámetros. Adam a menudo converge más rápidamente y con mayor estabilidad.
\end{resbox}
\end{frame}

\subsection{Backpropagation: Cálculo Vectorial del Gradiente}
\begin{frame}[allowframebreaks]{Backpropagation: Cálculo Vectorial del Gradiente}
\justifying
La retropropagación es una aplicación eficiente de la regla de la cadena para calcular el gradiente en el espacio de parámetros.

\begin{equation}
    \frac{\partial J}{\partial \theta^{(l)}_{i,j}} = \sum_{k} \frac{\partial J}{\partial z^{(l+1)}_k} \frac{\partial z^{(l+1)}_k}{\partial \theta^{(l)}_{i,j}}
\end{equation}

Donde \(\theta^{(l)}_{i,j}\) es un parámetro en la capa \(l\), y \(z^{(l+1)}\) es la salida de la capa \(l+1\). Cada término en la suma es un producto escalar de vectores.


%🏈
\end{frame}
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 5.5

\begin{frame}[allowframebreaks]{Figura 5.5: Flujo de Gradientes en la Retropropagación.}
\begin{figure}
   \centering
    \includegraphics[width= 0.9 \textwidth]{Figura_5_5.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 5.5
\end{frame}

\begin{frame}[allowframebreaks]{Actualización de Parámetros como Operación Vectorial}
%🏈

\begin{resbox}
\textbf{Figura 5.5: Flujo de Gradientes en la Retropropagación.} Los gradientes fluyen desde la capa de salida hacia atrás a través de la red, calculando las derivadas parciales en cada capa. Esto es un proceso vectorial a través del grafo computacional.
\end{resbox}
\end{frame}

\section{De la Clasificación a la Detección: El Espacio Vectorial de YOLO}
\begin{frame}{De la Clasificación a la Detección: El Espacio Vectorial de YOLO}
\justifying
\begin{chapterbox}{Panorama del capítulo}
La comprensión de la CNN como un sistema de transformaciones vectoriales es el fundamento para entender modelos de detección de objetos como YOLO.
\end{chapterbox}
\end{frame}

\subsection{La CNN como Backbone Vectorial}
\begin{frame}[allowframebreaks]{La CNN como Backbone Vectorial}
\justifying
En YOLO, la CNN (el backbone) transforma la imagen de entrada en un mapa de características que codifica información espacial y semántica.

\begin{equation}
    \mathbf{X}_{\text{imagen}} \in \mathbb{R}^{3 \times H \times W} \xrightarrow{\text{Backbone CNN}} \mathbf{F} \in \mathbb{R}^{C \times H' \times W'}
\end{equation}

Donde \(\mathbf{F}\) es el mapa de características, una representación vectorial de la imagen en un espacio de características donde cada "celda" contiene información sobre la región correspondiente de la imagen.
\end{frame}

\subsection{Predicciones de YOLO como Vectores}
\begin{frame}[allowframebreaks]{Predicciones de YOLO como Vectores}
\justifying
Para cada celda \(c\) en el mapa de características, YOLO predice:

\begin{itemize}
    \item \(p_{\text{obj}}\): Probabilidad de presencia de objeto (escalar)
    \item \((t_x, t_y, t_w, t_h)\): Coordenadas de la caja delimitadora (vector en \(\mathbb{R}^4\))
    \item \(\mathbf{p}_{\text{class}}\): Probabilidades de clase (vector en \(\mathbb{R}^{10}\))
\end{itemize}

La salida de YOLO es un tensor \(\mathbf{Y} \in \mathbb{R}^{H' \times W' \times (5 + 10)}\), donde cada celda produce un vector en \(\mathbb{R}^{15}\). La pérdida de YOLO combina términos de clasificación y regresión:

\begin{align}
  J_{\text{YOLO}}
  &= \lambda_{\text{coord}}\sum_i\sum_j
     \mathbf{1}_{ij}^{\text{obj}}
     \left\lVert \operatorname{box}_{ij}
     -\widehat{\operatorname{box}}_{ij}\right\rVert^2 \notag\\
  &\quad+\sum_i\sum_j\mathbf{1}_{ij}^{\text{obj}}
     \left\lVert\mathbf{p}_{ij}
     -\widehat{\mathbf{p}}_{ij}\right\rVert^2+\cdots .
\end{align}

%🏈
\end{frame}
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 5.5

\begin{frame}[allowframebreaks]{Figura 6.1: YOLO en el Espacio Vectorial.}
\begin{figure}
   \centering
    \includegraphics[width= 0.9 \textwidth]{Figura_6_1.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 5.5
\end{frame}

\begin{frame}[allowframebreaks]{YOLO en el Espacio Vectorial}
%🏈

\begin{resbox}
\textbf{Figura 6.1: YOLO en el Espacio Vectorial.} La imagen se transforma en un mapa de características. Cada celda de este mapa es un vector que codifica información sobre la presencia de objetos, su ubicación y su clase.
\end{resbox}
\end{frame}

\subsection{El Espacio Vectorial de las Cajas Delimitadoras}
\begin{frame}[allowframebreaks]{El Espacio Vectorial de las Cajas Delimitadoras}
\justifying
Las cajas delimitadoras se representan como vectores en \(\mathbb{R}^4\): \((x, y, w, h)\) o \((t_x, t_y, t_w, t_h)\) en YOLO. La pérdida de localización es una norma en este espacio, típicamente la norma L2 (error cuadrático medio) o una combinación de pérdidas.

\begin{equation}
    \text{BoxLoss} = \| \mathbf{b} - \hat{\mathbf{b}} \|^2 = (x - \hat{x})^2 + (y - \hat{y})^2 + (w - \hat{w})^2 + (h - \hat{h})^2
\end{equation}

%🏈
\end{frame}
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 5.5

\begin{frame}[allowframebreaks]{Figura 6.2: Espacio de las Cajas Delimitadoras.}
\begin{figure}
   \centering
    \includegraphics[width= 0.80 \textwidth]{Figura_6_2.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 5.5
\end{frame}

\begin{frame}[allowframebreaks]{YOLO en el Espacio Vectorial}
%🏈

\begin{resbox}
\textbf{Figura 6.2: Espacio de las Cajas Delimitadoras.} Cada caja es un punto en \(\mathbb{R}^4\). La pérdida de localización mide la distancia entre la caja predicha y la caja real en este espacio.
\end{resbox}
\end{frame}

\subsection{Conclusión}
\begin{frame}[allowframebreaks]{Conclusión}
\justifying
El análisis vectorial y matricial del notebook \texttt{cnn\_visdrone\_paso\_paso.ipynb} revela la estructura matemática subyacente de las CNNs. Cada operación, desde la preparación de datos hasta la optimización, es una transformación en un espacio vectorial de alta dimensión. Esta comprensión es esencial para:

\begin{enumerate}
    \item \textbf{Depuración e Interpretación:} Entender por qué una red se comporta de cierta manera.
    \item \textbf{Diseño de Arquitecturas:} Crear nuevas arquitecturas basadas en principios de álgebra lineal.
    \item \textbf{Transición a Modelos Avanzados:} Aplicar el mismo marco vectorial para comprender YOLO y otros modelos de detección de objetos.
\end{enumerate}

La capacidad de PyTorch para operar en estos espacios vectoriales de manera eficiente y diferenciable es la clave que permite el desarrollo de modelos de aprendizaje profundo a gran escala.

%🏈
\end{frame}
%✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ Figura 5.5

\begin{frame}[allowframebreaks]{Figura 6.3: Del Espacio de Características a la Detección.}
\begin{figure}
   \centering
    \includegraphics[width= 0.9 \textwidth]{Figura_6_3.png}
%   \caption{ 1.1 Representación Vectorial de una Imagen}
%    \label{fig:generative_models}
\end{figure}

%🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴✅Figura 5.5
\end{frame}

\begin{frame}[allowframebreaks]{YOLO en el Espacio Vectorial}
%🏈


\begin{resbox}
\textbf{Figura 6.3: Del Espacio de Características a la Detección.} El espacio de características aprendido por la CNN se utiliza como base para la detección de objetos. Cada celda del mapa de características produce un vector de predicción que codifica la presencia y ubicación de objetos.
\end{resbox}
\end{frame}

\end{document}
```
