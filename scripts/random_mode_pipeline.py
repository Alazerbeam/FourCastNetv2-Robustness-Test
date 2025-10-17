import os
from generate_forecast import run_inference
from compute_error import generate_global_error_dataset
from visualize_forecast import animate_frames, visualize_global
from plot_errors import plot_pixelwise_error_hists, plot_pixelwise_error_summary, plot_pixelwise_error_moments
from config import *

for distribution in DISTRIBUTIONS:  # repeat for each distribution
    if PREDICT:
        print(f"Generating forecast from {distribution} initial conditions...")
        run_inference(mode="random", distribution=distribution, verbose=True)

    if VISUALIZE_GLOBAL:
        plot_dir = os.path.join(GLOBAL_PRED_PLOT_DIR, "random", distribution)
        print(f"Plotting forecast in {plot_dir}...")
        for channel in CHANNELS:
            visualize_global(
                plot_dir, PRED_PATH, channel, TIMESTEPS, ERA5_STATS_DIR, 
                title_prefix = f"FourCastNetv2 prediction from {distribution} - {channel}", 
                filename_prefix = f"{channel}_prediction_{distribution}"
            )
            animate_frames(os.path.join(plot_dir, channel), animation_name=f"animated_prediction_{channel}_{distribution}.gif")

    if PLOT_ERROR_GLOBAL:
        generate_global_error_dataset(ERROR_DATAPATH, TRUE_PATH, PRED_PATH)
        plot_dir = os.path.join(GLOBAL_ERR_PLOT_DIR, "random", distribution)
        print(f"Plotting error in {plot_dir}...")
        for channel in CHANNELS:
            visualize_global(
                plot_dir, ERROR_DATAPATH, channel, TIMESTEPS, ERA5_STATS_DIR, 
                title_prefix = f"FourCastNetv2 Hurricane Florence Prediction Error - {distribution} distribution - {channel}", 
                filename_prefix = f"{channel}_error_{distribution}",
                is_error_plot = True
            )
            animate_frames(os.path.join(plot_dir, channel), animation_name=f"animated_error_{channel}_{distribution}.gif")
            plot_pixelwise_error_hists(
                os.path.join(plot_dir, channel), ERROR_DATAPATH, channel, TIMESTEPS,
                title = f"Global {channel} error histogram - {distribution}",
                filename_prefix = f"{channel}_error_hist_{distribution}"
            )
            plot_pixelwise_error_summary(
                os.path.join(plot_dir, channel), ERROR_DATAPATH, channel, TIMESTEPS, 
                title = f"Global {channel} error summary - {distribution}", 
                filename = f"{channel}_error_summary_{distribution}"
            )
            plot_pixelwise_error_moments(
                os.path.join(plot_dir, channel), ERROR_DATAPATH, channel, TIMESTEPS, 
                title = distribution, 
                filename = f"Global {channel} error moments - {distribution}"
            )

if CLEAN_UP:
    if os.path.exists(PRED_PATH):
        print("Deleting forecast...")
        os.remove(PRED_PATH)
    if os.path.exists(ERROR_DATAPATH):
        print("Deleting error dataset...")
        os.remove(ERROR_DATAPATH)
    