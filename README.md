# Beyond Hubble: Cepheid Calibration & The Cosmic Distance Ladder
### Astroclub Winter Project 2025-26

Beyond Hubble is an astrophysics and machine learning project focused on calibrating the **Cosmic Distance Ladder** using **Cepheid Variable Stars** from the Gaia DR3 survey. The project implements a two-stage pipeline:
1. **Unsupervised Classification**: Using light-curve statistics (variance, skewness, kurtosis) and KMeans clustering to naturally separate Cepheid variables from background stars without using labels.
2. **Robust Regression**: Fitting the **Leavitt Law** (Period-Luminosity relation) using a RANSAC regressor to calculate stellar distances while automatically filtering out measurement noise and non-Cepheid outliers.

---

## Table of Contents
1. [Astrophysics Theoretical Foundation](#astrophysics-theoretical-foundation)
   - [The Cosmic Distance Ladder](#the-cosmic-distance-ladder)
   - [Cepheid Variables & The Leavitt Law](#cepheid-variables--the-leavitt-law)
   - [Hubble's Law & Redshift Expansion](#hubbles-law--redshift-expansion)
2. [Machine Learning & Analysis Pipeline](#machine-learning--analysis-pipeline)
   - [Dataset Overview (Gaia DR3)](#dataset-overview-gaia-dr3)
   - [Step 1: Unsupervised KMeans Classification](#step-1-unsupervised-kmeans-classification)
   - [Step 2: Robust Period-Luminosity Fitting (RANSAC)](#step-2-robust-period-luminosity-fitting-ransac)
3. [Key Insights & Performance](#key-insights--performance)
4. [Repository Structure](#repository-structure)
5. [Installation & Usage](#installation--usage)

---

## Astrophysics Theoretical Foundation

### The Cosmic Distance Ladder
Astronomers cannot measure vast distances in space using a single method. Instead, they rely on the **Cosmic Distance Ladder**, a series of overlapping techniques where each "rung" is calibrated using the previous, more local rung:
1. **Radar Ranging**: Measuring planetary distances in the Solar System by timing radio waves reflected off planets:
   $$R = \frac{c \cdot \Delta t}{2}$$
2. **Trigonometric Parallax**: Measuring nearby stellar distances by observing how their positions shift relative to background stars as Earth orbits the Sun. The unit **parsec (pc)** is defined as the distance where a shift of 1 astronomical unit (AU) subtends an arcsecond ($1/3600^\circ$):
   $$d = \frac{1}{p} \quad [\text{pc}]$$
3. **Main Sequence Fitting**: Calibrating star cluster distances by vertically matching their observed Color-Magnitude Diagrams (CMD) to a standard Zero-Age Main Sequence (ZAMS) template using the distance modulus:
   $$m - M = 5 \log_{10}(d) - 5$$
4. **Cepheid Variables (Standard Candles)**: Calibrating distances using pulsating stars that follow a strict relationship between their pulsation periods and intrinsic absolute magnitudes.
5. **Hubble-Lemaître Law**: Using the recessional velocity of distant galaxies to measure cosmological expansion.

### Cepheid Variables & The Leavitt Law
Cepheid variables are pulsating stars whose diameters change periodically, causing their apparent brightness to fluctuate in a predictable, repeating cycle. 
In 1912, Henrietta Swan Leavitt discovered that a Cepheid’s pulsation period ($P$) is directly related to its absolute magnitude ($M$). This is known as the **Leavitt Law**:
$$M = a \log_{10}(P) + b$$

Where:
* $P$ is the pulsation period in days.
* $M$ is the absolute magnitude (intrinsic brightness).
* $a$ (slope) and $b$ (intercept) are coefficients that vary slightly depending on the waveband (e.g. Classical vs. Type II Cepheids).

By measuring the period $P$ (from light curves) and apparent magnitude $m$, we can compute absolute magnitude $M$, which allows us to solve for distance $d$:
$$d = 10^{\frac{m - M + 5}{5}} \quad [\text{pc}]$$

### Hubble's Law & Redshift Expansion
At cosmological scales, Edwin Hubble observed that galaxies are moving away from Earth at recessional velocities ($v$) proportional to their distance ($d$):
$$v = H_0 \cdot d$$

Where $H_0$ is the **Hubble Constant** (typically $\approx 70 \text{ km/s/Mpc}$). Recessional velocity is derived from **redshift ($z$)**, the stretching of light wavelengths due to the Doppler expansion of space:
$$z = \frac{\lambda_{\text{observed}} - \lambda_{\text{rest}}}{\lambda_{\text{rest}}}$$
For nearby galaxies ($z \le 0.1$), recessional velocity is approximated linearly:
$$v \approx c \cdot z$$

---

## Machine Learning & Analysis Pipeline

### Dataset Overview (Gaia DR3)
The dataset comprises stellar measurements from the Gaia DR3 survey structured into batches (clusters). Key variables include:
* **Astrophysical Measurements**: Right ascension (`ra`), declination (`dec`), true distance (`dist`), period (`period`), apparent magnitude (`apparent_mag`).
* **Light Curve Statistics**: `variance`, `skew`, and `kurt` (kurtosis).
* **Target Classification**: `is_cepheid` (used only for evaluation).

### Step 1: Unsupervised KMeans Classification
To demonstrate that Cepheid variables naturally group separate from common stars without supervision, the pipeline uses **KMeans Clustering**:
1. **Feature Selection**: Selects `skew` and `kurt` of the stellar light curves as classification features.
2. **Preprocessing**: standardizes features with `StandardScaler` to ensure equivalent feature weighting.
3. **Clustering ($K=14$)**: Splits the training subset into 14 distinct clusters.
4. **Heuristic Identification**: Cepheid variables are rare, highly variable stars that exhibit significant dispersion in the feature space compared to standard stars. By computing the distances of points inside each cluster to their centroid, the cluster with the **maximum distance variance** is identified as the Cepheid cluster:
   $$\text{Variances}_k = \frac{1}{N_k}\sum_{i=1}^{N_k} \|x_i - \mu_k\|^2$$
   $$\text{Cepheid Label} = \text{argmax}(\text{Variances}_k)$$
5. **Mapping & Scoring**: The identified cluster receives predicted label `1` (Cepheid), and all other clusters are mapped to `0`. A confusion matrix and classification report are then evaluated.

### Step 2: Robust Period-Luminosity Fitting (RANSAC)
Once the Cepheids are identified, they are used to fit the Leavitt Law.
1. **Feature Formulation**:
   - $X$: Logarithm of period ($\log_{10}(P)$) and apparent magnitude ($apparent\_mag$).
   - $Y$: Logarithm of true distance ($\log_{10}(dist)$).
2. **RANSAC Regression**: Standard Linear Regression is highly sensitive to outliers (e.g. false Cepheids from KMeans errors). To resolve this, a **RANSAC (Random Sample Consensus) Regressor** is trained:
   - It repeatedly selects random subsets of stars to fit linear equations.
   - It counts how many total stars match the model within a threshold (inliers).
   - It outputs the final linear coefficients $(w_1, w_2)$ and intercept $(b)$ using the best fit inlier plane:
     $$\log_{10}(\text{dist}) = w_1 \log_{10}(P) + w_2 (\text{apparent\_mag}) + b$$
3. **3D Visualization**: Renders a 3D scatter plot of the data points color-coded by inlier/outlier status along with the fitted regression plane.

---

## Key Insights & Performance

* **KMeans Clustering**: The unsupervised maximum cluster variance heuristic proves highly effective at identifying the Cepheid variable subclass. It successfully isolates them based on the extreme values of skewness and kurtosis that result from periodic stellar pulsations.
* **Robust RANSAC Fitting**: The RANSAC regressor effectively mitigates classification mistakes from the KMeans stage. It excludes false positive outliers that lie off the Period-Luminosity-apparent magnitude plane, generating highly accurate coefficients for distance estimations.

---

## Repository Structure

```
Beyond-Hubble/
├── .gitignore             # Git ignore configuration
├── README.md              # Project documentation (this file)
├── requirements.txt       # Python dependencies
├── data/                  # Star cluster datasets (place beyond_hubble.csv here)
├── notebooks/             # Principal research and analysis notebooks
│   └── beyond_hubble.ipynb
└── reports/               # Full scientific reports and project guides
    └── Beyond Hubble.pdf
```

---

## Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amanpal27/Beyond-Hubble.git
   cd Beyond-Hubble
   ```

2. **Set up virtual environment & install requirements**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Set up Dataset**:
   Place the dataset `beyond_hubble.csv` inside the `data/` folder (or ensure it is located in the root of the project as referenced in the notebook).

4. **Running the Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/beyond_hubble.ipynb
   ```
