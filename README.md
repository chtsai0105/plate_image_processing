# Autocrop
A opencv based python script that used to crop out the 96-well colonies on a big agar plate.

## Usage
Use `python autocrop.py --help` to see more details.
```
positional arguments:
  image                 Image that contains colony plates.

options:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        Output directory.
  -H, --high            Use parameters for high growth plate
  -q, --quiet           Quiet mode to suppress msg.
  -v, --verbose         Verbose mode for debug.
```

Let's process the image stored in `original_images/3_1_30.jpg` and output to `cropped_images`.
```
python autocrop.py original_images/3_1_30.jpg -o cropped_images
```
The default parameters is mainly for low/medium groth rate plates.
Processing high growth rate plates with default parameters sometimes will encounter errors.
```
python autocrop.py original_images/10_1_30C.jpg -o cropped_images
```
```
2023-09-11 15:51:36,025 autocrop INFO [10_1_30C.jpg] Start processing original_images/10_1_30C.jpg
2023-09-11 15:51:40,858 autocrop INFO [10_1_30C.jpg] Got image that contains markers only.
2023-09-11 15:51:40,863 autocrop INFO [10_1_30C.jpg] Write marker image to cropped_images/10_1_30C_M.jpg
2023-09-11 15:51:40,863 autocrop INFO [10_1_30C.jpg] Looking for plates in the container image
2023-09-11 15:51:43,974 autocrop INFO [10_1_30C.jpg] Finding for plate that the ratio falls between 0.1-0.2 ...
2023-09-11 15:51:43,974 autocrop WARNING [10_1_30C.jpg] Cannot found any contour which area falls between 0.1-0.2.
2023-09-11 15:51:43,974 autocrop INFO [10_1_30C.jpg] Remove already generated files.
```
If the run fail to crop out any plate image, the already cropped markers image will also be removed.
In this case, you can try swtiching on the option `-H` to use parameters for high growth rate plate.
```
python autocrop.py original_images/10_1_30C.jpg -o cropped_images -H
```

### Process the entire folder
You can simply use the bash for loop to process the all the images contains in a folder.
```
for file in original_images/*; do python autocrop.py $i -o cropped_images; done
```
Since the default parameters are not going to work well in some of the high growth rate plates,
the below is the recommended processing step:
1. First run the for loop with the default parameters.
2. Manually check the images that are failed and delete them.
3. Run the for loop again with the `-H` option.

## Requirements
- Python >= 3.7
- opencv >= 4.8.0
- largestinteriorrectangle >= 0.2.0
- timeout-decorator >= 0.5.0

Use the environment.yml to install all the required packages:
```
conda env create -f environment.yml
```
