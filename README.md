# Perception-LOD: Beyond Fidelity — Rethinking Realism Through Human Behavior in VR City Modeling LOD Design

This repository contains the **code and resources** for the paper:  
> **"Beyond Fidelity: Rethinking Realism Through Human Behavior in VR City Modeling LOD Design"**

---

## Overview

Perception-LOD is a research project exploring how human visual perception and behavior can inform Level-of-Detail (LOD) design in virtual reality (VR) city modeling. By integrating eye-tracking technology with adaptive rendering techniques, this work aims to optimize visual quality and performance in immersive environments.

---

## Repository Structure

```
Perception-LOD/
├── Assets/                      # Unity project assets
│   ├── Scripts/                 # C# scripts for VR and eye-tracking
│   ├── Scenes/                  # Unity scenes
│   ├── Prefabs/                 # Prefabricated objects
│   ├── Materials/               # Material assets
│   ├── Textures/                # Texture assets
│   └── XR/                      # XR-related configurations
├── AnalysisScripts/             # Python analysis scripts
│   ├── PerceptionLOD.py         # Main perception analysis & clustering
│   ├── AOI.py                   # Area-of-interest analysis
│   ├── BestLOD.py               # LOD optimization algorithms
│   ├── HeatMap.py               # Heatmap visualization
│   └── SVM.py                   # SVM-based classification
├── ProjectSettings/             # Unity project settings
├── Packages/                    # Unity package configurations
├── Samples/                     # Sample data and outputs
└── Scenes/                      # Scene reference images
```

---

## Unity VR Project

### Requirements
- **Unity Engine**: 2021.3.33f1c1 (LTS)
- **Platform**: PICO VR (with eye-tracking support)
- **SDK**: PICO Unity Integration SDK

### Key Scripts
| Script | Description |
|--------|-------------|
| `EyeTrackingRecorder.cs` | Captures and records eye-gaze data from PICO headset, including world coordinates, UV mapping, and object hit detection |
| `ChangeImageAndSaveData.cs` | Manages scene transitions and data collection workflow |
| `DataExporter.cs` | Exports collected experimental data to CSV format |
| `testrototer.cs` | Utility script for testing and debugging |

### How to Use
1. Clone or download this repository
2. Open the project in **Unity Hub** with Unity 2021.3.33f1c1
3. Ensure PICO Unity Integration SDK is properly installed
4. Open the main scene from `Assets/Scenes/`
5. Connect a PICO VR headset with eye-tracking capability
6. Enter Play Mode or build and deploy to the device

---

## Analysis Scripts

The `AnalysisScripts/` directory contains Python scripts for processing and analyzing the collected eye-tracking data.

### Requirements
- Python 3.7+
- pandas, numpy, scikit-learn, matplotlib, scipy

### Scripts Overview
| Script | Functionality |
|--------|---------------|
| `PerceptionLOD.py` | K-means clustering of perception patterns, feature calculation (target score, roaming score, information entropy), 3D visualization |
| `AOI.py` | Area-of-Interest (AOI) analysis and statistical processing |
| `BestLOD.py` | Algorithms for determining optimal LOD configurations based on perceptual data |
| `HeatMap.py` | Generates visual heatmaps of gaze distribution |
| `SVM.py` | Support Vector Machine classification for perception patterns |

---

## Sample Data

Sample eye-tracking data and analysis outputs are located in the `Samples/` directory.

- **Format**: CSV files containing gaze coordinates, timestamps, object tags, and LOD levels
- **License**: For non-commercial research use only. Anonymized data — do not attempt re-identification of participants.

---

## Project Settings

- **Unity Version**: 2021.3.33f1c1
- **Render Pipeline**: URP (Universal Render Pipeline)
- **XR SDK**: PICO Unity Integration
- **Target Platform**: Android (PICO devices)

---

## Release

**v1.1** – Initial release with complete Unity project and analysis scripts

---

## Contact

**Wufan Zhao**  
Email: [wufanzhao@hkust-gz.edu.cn](mailto:wufanzhao@hkust-gz.edu.cn)  
Institution: The Hong Kong University of Science and Technology (Guangzhou)

---

## License

- **Code**: [MIT License](LICENSE) — see `LICENSE` file for details  
- **Sample Data**: Non-commercial research use only; anonymized; no re-identification allowed.

---

## Acknowledgements

We sincerely thank the editors and anonymous reviewers at **JAG** for their constructive and insightful feedback that significantly improved this work.

---

*Maintained by @ HKUST(GZ)*