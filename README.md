# nitk Neuroimaging Toolkit (nitk), and usage examples.

Nitk provides:

1. A template of a typical data science project following [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
- **nitk** aims to provides functionalities not (yet) available in [nilearn](https://nilearn.github.io/) or scikit learn. Alway prefers nilearn functionnality if available.
- **scripts** provides example usage.


Provide utils functions and commands organized by software.

        nitk/
        ├── bids
        │   ├── Related to BIDS
        ├── cat12vbm
        │   └── Related to CAT12 VBM pipeline
        ├── fs
        │   └── Freesufer
        ├── image
        │   └─── Image, referential, mask, resampling, preprocessing
        │        * (list of) Niftii image(s).
        |        * array data structure (n_subjects, n_channels, image_axis0, image_axis1, ...)
        ├── mesh
        │   └── Related to meshes (giftii)
        │        
        └── spmsegment
            └── Related to spm sgementation


## Data organisation



## Environement Creation for the first time

```
pixi init
pixi add pandas scikit-learn seaborn openpyxl ipykernel  nilearn fsleyes  shap
pixi shell
```

## Cloning this environement
```

pixi install
```

## Provides

- `niml`: library with basics utils functions
- `scripts_tables`: machine learning scripts using tables (csv/xlsx files)

