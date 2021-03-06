# Calibrating a Latent Order Book Model to Market Data
## LOB Evolution
![LOB Evolution](https://github.com/GantZA/CALOBMTMD/blob/main/scripts/price_paths/LOB.gif)
## SLOB Evolution
![SLOB Evolution](https://github.com/GantZA/CALOBMTMD/blob/main/scripts/price_paths/SLOB.gif)

# Overview

This repository holds the Julia 1.5 scripts used in the minor disseration, Calibrating a Latent Order Book Model to Market Data, by Michael Gant and supervised by Tim Gebbie as a part of the Masters in Advanced Analytics and Decision Sciences in the Statistical Sciences department at the University of Cape Town.

The code was run using the facilities provided by the UCT High Performance Computing centre. The code run on the HPC requires installing 2 Julia packages from GitHub, namely: SequentialLOB.jl, AdaptiveABC.jl. The Project.toml in this repo contains the required Julia dependancies. 

Generating the figures found in the disseration requires additional plotting libraries, one of which needs to be installed from GitHub, StylizedFacts.jl.

The real market data used to calibrate the LOB and SLOB models can be downloaded from https://data.mendeley.com/datasets/nt8nw28h7c/1 

# Installation

Using the Manifest.toml in this repo, you can create a new Julia environment which will install all the packages at the correct versions.

# Scripts Overview

The Julia scripts are stored within the scipts folder and are separated by their usecase. 

## Calibration Results and Figures

There are 2 calibration subfolders, `synthetic_calibration` and `market_data_calibration`. Within each are the folders pertaining to the different models, LOB, SLOB or ARMA. The results are generated and saved by one script, and then the `_results.jl` suffixed script will read the results and generate and save various plots.

## Numerical Solution Surface Plot

In the `numerical_solution` folder, the script `numerical_sol_figures.jl` will simulate and store the latent order book density values and then plot them with the interative plotting functionality from PyPlot.jl.

## Price Path Figures

Example mid-price path and log return plots are generated by the script in the `price_paths` folder.

## Stylised Facts

The stylised fact plots for the Market data, LOB simulated data and SLOB simulated data are generated by the script in the `stylised_facts` folder. 

## ARMA Free-Parameter Surfact Plot

A surface plot of 2 free-parameters (Phi (??) and Theta (??)) from the ARMA model with the MADWE distance function. 