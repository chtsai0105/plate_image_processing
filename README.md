# Autocrop

A opencv based python script that used to crop out the 96-well colonies on a petri dish.

## Usage

Use `python AUTOCROP --help` to see more details.

```
positional arguments:
  image                 Image that contains colony plates.

options:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        Output directory.
  -n N_PLATES, --n_plates N_PLATES
                        Number of plates on the petri dish. (Default=3)
  -p {H,M,L}, --param {H,M,L}
                        Force to use parameters for high/medium/low growing plate.
  -q, --quiet           Quiet mode to suppress msg.
  -v, --verbose         Verbose mode for debug.
```

Let's process the image stored in `original_images/3_1_30.jpg` and output to `cropped_images`.

```
python AUTOCROP original_images/3_1_30.jpg -o cropped_images
```

By default, AUTOCROP will automatically find a proper parameters for high/medium/low growing plates.
This image was detected to be a medium growing plate.

```
2023-09-26 14:59:03,719 autocrop INFO [3_1_30.jpg] Start processing original_images/3_1_30.jpg
2023-09-26 14:59:05,401 autocrop INFO [3_1_30.jpg] Use "M" parameter set
2023-09-26 14:59:05,517 autocrop INFO [3_1_30.jpg] Write adjusted image to output/3_1_30.jpg
2023-09-26 14:59:06,028 autocrop INFO [3_1_30.jpg] Write plate1 image to output/3_1_30_1.jpg
2023-09-26 14:59:06,045 autocrop INFO [3_1_30.jpg] Write plate2 image to output/3_1_30_2.jpg
2023-09-26 14:59:06,061 autocrop INFO [3_1_30.jpg] Write plate3 image to output/3_1_30_3.jpg
2023-09-26 14:59:06,064 autocrop INFO [3_1_30.jpg] Write marker image to output/3_1_30_M.jpg
```

However, the parameter auto-detection sometime will failed.

```
python AUTOCROP original_images/7_2_low_N.jpg -o cropped_images
```

```
2023-09-26 15:00:49,116 autocrop INFO [7_2_low_N.jpg] Start processing original_images/7_2_low_N.jpg
2023-09-26 15:00:51,092 autocrop INFO [7_2_low_N.jpg] Use "H" parameter set
2023-09-26 15:00:51,207 autocrop INFO [7_2_low_N.jpg] Write adjusted image to output/7_2_low_N.jpg
2023-09-26 15:00:51,285 autocrop WARNING [7_2_low_N.jpg] Cannot found proper dilation parameter. Aborted.
```

In this case, you can force to use a particular parameter set by specifying `-p [H/M/L]`. E.g.

```
python AUTOCROP original_images/7_2_low_N.jpg -o cropped_images -p M
```

Sometimes the issue would be resolved.

```
2023-09-26 15:01:46,532 autocrop INFO [7_2_low_N.jpg] Start processing original_images/7_2_low_N.jpg
2023-09-26 15:01:48,468 autocrop INFO [7_2_low_N.jpg] Use "M" parameter set
2023-09-26 15:01:48,583 autocrop INFO [7_2_low_N.jpg] Write adjusted image to output/7_2_low_N.jpg
2023-09-26 15:01:48,941 autocrop INFO [7_2_low_N.jpg] Write plate1 image to output/7_2_low_N_1.jpg
2023-09-26 15:01:48,956 autocrop INFO [7_2_low_N.jpg] Write plate2 image to output/7_2_low_N_2.jpg
2023-09-26 15:01:48,970 autocrop INFO [7_2_low_N.jpg] Write plate3 image to output/7_2_low_N_3.jpg
2023-09-26 15:01:48,977 autocrop INFO [7_2_low_N.jpg] Write marker image to output/7_2_low_N_M.jpg
```

### Process the entire folder

You can simply use the bash for loop to process the all the images contains in a folder.

```
for file in original_images/*; do python AUTOCROP $i -o cropped_images; done
```

Since the parameter auto-detection doesn't work well on some of the plates,
the below is the recommended processing step:

1. Since most of the plates are considered to be medium growing plate. We first run the for loop with the `-m M` option.
2. Manually check the images that are failed and delete them.
3. Run the for loop again with the `-m L` option. Delete the failed images.
4. Run the for loop again with the `-m H` option. Delete the failed images.
5. Manually crop image by [GIMP](https://www.gimp.org/)

### Try out on the notebook

A jupyter notebook `test.ipynb` was also uploaded to the repo. You can try to fine-tune the program by yourself.
Please keep the lines importing the library to make it works. Especially the line which import the autocrop functions.

```
import matplotlib.pyplot as plt
from types import SimpleNamespace

from autocrop.autocrop import *
```

## Requirements

- Python >= 3.7
- opencv >= 4.8.0
- largestinteriorrectangle >= 0.2.0
- timeout-decorator >= 0.5.0

Use the environment.yml to install all the required packages:

```
conda env create -f environment.yml
```
