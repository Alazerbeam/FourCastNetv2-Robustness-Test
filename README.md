# SDSU Robustness Test for FourCastNetv2 AI Weather Forecast Models
This repository contains the supporting materials for my paper entitled "Robustness Test for AI Forecasting of Hurricane Florence Using FourCastNetv2 and Random Perturbations of the Initial Condition" authored by Adam Lizerbram, Shane Stevenson, Iman Khadir, Matthew Tu, and Samuel S. P. Shen.

# Summary of this Project
This project aims to explore the robustness of NVIDIA's AI weather forecasting model, FourCastNetv2 (FCNv2), to randomness added to the initial conditions. We use FCNv2 to track Hurricane Florence under varying amounts of Gaussian noise added to the initial condition (September 13, 2018 at 00:00 UTC). We also use FCNv2 to generate forecasts under fully random initial conditions sampled from normal, lognormal, chi square, and uniform distributions. We found that FCNv2 is able to generate accurate hurricane tracks under moderate input noise, and even under high input noise, still follows the correct direction trend, although positional accuracy degrades. Even under fully random initial conditions, FCNv2 is able to produce smooth forecasts. 

# Sample Figures for this Project
See the sample_figures folder for sample figures generated for this project. 

# Computer Code for Reproduction for the Figures and Animations of the Paper
See the scripts folder for the code used for this research project. The scripts plot_errors.py and visualize_forecast.py contain the functions used to generate the plots.

# Data Needed for the Reproduction of the Figures
The data to reproduce these figures is too large to store on GitHub. Here is a link to them in Google Drive: [link].

# Computer Code for this Research Project
In the scripts folder, the files noise_mode_pipeline.py and random_mode_pipeline.py call the functions in the other scripts to generate all forecasts and figures automatically.

# Contact Me
If you have any questions about this project, contact my at adamlizerbram@gmail.com.

# References
