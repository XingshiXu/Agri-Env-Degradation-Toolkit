# Agri-Env-Degradation-Toolkit
Implementation of environmental degradation algorithms (Rainfall, Fog, Low-light) for robustness evaluation  

  
## 🌟 Key Features
**Rainfall Simulation:** Multi-factor coupled modeling including rain streaks, lens raindrops, and atmospheric scattering.

**Fog Simulation:** Physically-based degradation using the Koschmieder atmospheric scattering model.

**Low-light Simulation:** Illumination adjustment using Gamma correction and global attenuation.

**Intensity Control:** Each degradation supports continuous parameter adjustment across multiple severity levels.  


### 1. Rainfall Modeling
Controlled by three joint parameters:
- **$\alpha$ (Brightness):** $\{0.92, 0.84, 0.76, 0.68\}$
- **$N$ (Rain streaks):** $\{134, 379, 697, 1073\}$
- **$\eta$ (Fogging factor):** $\{0.04, 0.08, 0.12, 0.16\}$

### 2. Fog Simulation (Koschmieder Model)
Based on the equation: 
$$I(x) = J(x)t(x) + A(1 - t(x))$$
- **$\beta$ (Scattering coefficient):** $\{0.5, 1.0, 1.5, 2.0\}$

### 3. Low-light Simulation
Joint adjustment of Gamma ($\gamma$) and Attenuation ($\alpha$):
- **$(\gamma, \alpha)$:** $\{(1.2, 0.9), (1.5, 0.7), (2.0, 0.5), (2.5, 0.4)\}$

The construction of this warehouse draws on the production ideas and physical modeling methods of the following classic datasets in the field of computer vision:  
https://github.com/ZhangXinNan/RainDetectionAndRemoval  
https://github.com/Boyiliee/RESIDE-dataset-link  
https://github.com/zhangyhuaee/KinD  
