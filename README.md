# SDSU Robustness Test for FourCastNetv2 AI Weather Forecasting Model
This repository contains the supporting materials for my paper entitled "Robustness Test for AI Forecasting of Hurricane Florence Using FourCastNetv2 and Random Perturbations of the Initial Condition" authored by Adam Lizerbram, Shane Stevenson, Iman Khadir, Matthew Tu, and Samuel S. P. Shen.

# Summary of this Project
This project aims to explore the robustness of NVIDIA's AI weather forecasting model, FourCastNetv2 (FCNv2), to randomness added to the initial conditions. We use FCNv2 to track Hurricane Florence under varying amounts of Gaussian noise added to the initial condition (September 13, 2018 at 00:00 UTC). We also use FCNv2 to generate forecasts under fully random initial conditions sampled from normal, lognormal, chi square, and uniform distributions. We found that FCNv2 is able to generate accurate hurricane tracks under moderate input noise, and even under high input noise, still follows the correct direction trend, although positional accuracy degrades. Even under fully random initial conditions, FCNv2 is able to produce smooth forecasts. 

# Sample Figures for this Project
See the sample_figures and sample_animations folders for sample figures and animations generated for this project. 

# Computer Code for Reproduction for the Figures and Animations of the Paper
See the scripts folder for the code used for this research project. The scripts plot_errors.py and visualize_forecast.py contain the functions used to generate the plots.

# Data Needed for the Reproduction of the Figures
The data to reproduce these figures is too large to store on GitHub. Here is a link to them in Google Drive: https://drive.google.com/drive/folders/17KHU658QeK5rgRNxJkutsLPmwMibSkDm?usp=sharing.

# Computer Code for this Research Project
In the scripts folder, the files noise_mode_pipeline.py and random_mode_pipeline.py call the functions in the other scripts to generate all forecasts and figures automatically. Use config.py to define your directories and steps the pipeline should complete.

# Tutorial and Contact Me
See Setting_Up_Earth2MIP.pdf for a step-by-step guide to using Earth-2 MIP. I would recommend using a Linux-based system, since Windows can have errors working with NetCDF files. If you have any questions about this project, contact me at adamlizerbram@gmail.com.

# References
Hersbach, H., and Coauthors, 2023a: ERA5 hourly data on pressure levels from 1940 to present. Accessed: 2025-08-19, https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview, https://doi.org/10.24381/cds.bd0915c6.

Hersbach, H., and Coauthors, 2023b: ERA5 hourly data on single levels from 1940 to present. Accessed: 2025-08-19, https://cds.climate.copernicus.eu/datasets/reanalysis-era5-signle-levels?tab=overview, https://doi.org/10.24381/cds.adbb2d47.

NVIDIA, 2025: Earth-2 Model Intercomparison Project. GitHub, Version 0.2.0a0, https://github.com/NVIDIA/earth2mip.

Bonev, B., and Coauthors, 2023: Spherical Fourier neural operators: Learning stable dynamics on the sphere. arXiv Preprint, https://doi.org/10.48550/arXiv.2306.03838.
