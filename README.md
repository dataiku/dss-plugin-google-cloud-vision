# Google Cloud Vision plugin

This plugin provides leverages various Google Cloud AI APIs for Computer Vision.

## Pre-requisites

- A service account key from a GCP project with access to the Cloud Vision APIs
- The Arial TrueType font installed on the DSS server

## Recipes

The plugin offers a suite of recipes:

* **image labelling**: the recipe calls the [`label_detection`](https://cloud.google.com/vision/docs/detecting-labels#vision-label-detection-python) method of the Cloud Vision API and returns a list of inferred categories from items present in the input image. Each label inferred is defined by:
    - a description
    - a confidence score
    - a topicality score that reflects the relevance of the detected label compared to the overall content of the image
    - the machine-generated identifier (`mid`) of the label in the Google Knowledge Graph
* **object localization**: the recipe calls the [`object_localization`] method of the Cloud Vision API to detect object types and positions within a given image. For each input image, a list of objects is generated, containing:
    - the name of the object
    - the coordinates of a rectangle that defines the boundaries of the object within the image (bounding box)
    You can optionally generate an output folder that contains the images with the bounding bowes drawn on it.
* **draw bounding box**: the recipe takes as an input:
    - a folder containing images
    - a dataset containing the object localization information for each image in the folder
    On each image in the input folder it will draw the corresponding bounding boxes and store the new image in the specified output folder.


## Authentication

Accessing the Cloud API endpoints requires a *service account key* that you need to generate in your GCP project. Once this is done, you will have two options to enforce authentication using that key:

- **application default credentials**: on the DSS server you need to create an environment variable called `GOOGLE_APPLICATION_CREDENTIALS` which contains the absolute path to your service account key.
- **service account key**: manually enter the absolute path to the key within the plugin interface

## External resources

- [Cloud Vision AI doc](https://cloud.google.com/vision/docs/)

