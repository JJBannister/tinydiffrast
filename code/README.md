# TinyDiffRast Code

## Setup/Install
To run the tests and examples, it is recommended to use a virtual python environment https://docs.python.org/3/library/venv.html#module-venv. 
Run the following command to create a venv.

    $ python -m venv venv

Within a virtual environment, the tinydiffrast package can be installed with pip.
If you intend to edit the package, it is best to use a dev installation so that you do not need to re-install after every change.
https://setuptools.pypa.io/en/latest/userguide/development_mode.html 

Activate your venv and then run the following command from within the /code directory to install tinydiffrast in development mode:

    $ pip install -e .

After installation, you should be able to run both the tests and examples within the virtual enviroment where the tinydiffrast package was installed.