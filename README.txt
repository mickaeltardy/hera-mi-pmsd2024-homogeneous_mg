The Output folder contains files relevant for the following:

Breast Density Classification Results
TSNE plots for different processing pipelines in the recipes folder

The src folder contains the following contents:

Notebooks: 'Analysis' and 'Classification'

Analysis Notebook: Experiments with Image Analysis and Image processing and plots TSNE projections to visualize difference in clusters of the different vendors

Classification Notebook: Builds and tests a Breast Density Classification Network using 2 methods:

1. Using the model architecture from https://github.com/nyukat/breast_density_classifier
2. Using a custom architecture that borrows from the previously mentioned repository but with slight adjustments

The results from this notebook can be found in the breast density results folder in the Output folder.

For Image Processing:

The file process.py contains the processing pipeline which implements a separate series of processing steps for each vendor.
To use the pipeline, you simply need to import process_images() from the file and pass an image as input as well as it's vendor.

The current supported vendors are: Siemens, Planmed, IMS and DDSM.

Other recipes for process_images() can be found in the recipes folder in the Output folder which contains other combinations of processing functions for each vendor as well as their 
effect on the TSNE plots. 