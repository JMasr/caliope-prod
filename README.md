``# caliope-pytorch

Caliope Toolkit implemented in Pytorch

## Repository's environment setup using conda.
`$ conda env create --name caliope --file environment.yml`

`$ conda activate caliope`

`$ pip install -r requirements.txt`

## How to:

### From a cmd:

A default call just needs as an arguments:
 1. **-i** -> path to the directory with the **file.eaf** and the **wordconfid.txt**  
 2. **--language** -> two options **gl** or **es** (gl is the default value)

`(caliope):$ python elan2cvs2srt.py -i /path/to/file.eaf --language gl`