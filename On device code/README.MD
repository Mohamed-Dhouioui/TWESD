## Usage

1. Put the `run.py` file and the `Images` folder on the device:
2. Install the dependencies:
```shell
pip install tflite-runtime<=2.8.0
pip install pillow
pip install argparse
```
3. run the script:

```shell
python run.py -m <model_name> -i <image_path> -s <image_size>
```
For example:
```shell
python run.py -m "CNN model creation\int_quant_model.tflite" -i Images\Te-gl_0036.jpg -s 180