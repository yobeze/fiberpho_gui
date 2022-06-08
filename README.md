# CU Boulder Donaldson Lab: Fiber Photometry GUI

## Installation Instructions

**New Users: Follow the instructions below to install the GUI:**
> Our code utilizes Python and numerous packages within it, if you do not already have Python installed on your device, please refer to this link before continuing: https://wiki.python.org/moin/BeginnersGuide/Download

1. Download Code
- Click on the green button labeled "Code" located at the top right corner of this repository and click on Download ZIP (Ensure that this file is saved locally on your device, in other words not on any cloud environments.
- Go to the downloaded zip file on your device and place it somewhere convenient to you so that you may easily navigate to it, again avoiding cloud storage.
- Unzip the file by clicking on it or using an unzipping utility (e.g. WinRAR), and take note of the Fiber-Pho-Main folder's path, we will need this later.)
    - Mac/Unix: Right click folder &rarr Hold Option key &rarr Click Copy "Fiber-Pho-Main" as Pathname.
    - Windows: Right click folder &rarr Properties &rarr Take note of text next to "Location:"

2. Create Virtual Environment
    > We will be creating a virtual environment to ensure that there are no issues with any currently installed python packages/modules. It may seem tedious/long but it is for precautionary purposes. You may install the GUI using either Anaconda or PIP/PyPI, the instructions for both are below:
    - Using PIP/PyPI:
        1. Open a new terminal window and confirm that you are in your Home directory/folder
            - Mac/Unix: Type "cd ~"
            - Windows: Ensure the directory adjacent to the cursor looks like this - "C:\Users\your-username\"
        2. Execute the following command: `python3 -m pip install --user virtualenv`
        3. Navigate to the location of the FiberGui folder in the command line.
            - E.g. "cd Desktop/FiberPhoGui"
        4. Run the following command: 
            - Mac/Unix: `python3 -m venv gui_env`
            - Windows: `py -m venv gui_env`
        5. Activate the virtual environment:
            - Mac/Unix: `source gui_env/bin/activate`
            - Windows: `.\gui_env\Scripts\activate`
        6. Confirm you are in the virtual environment:
            - Mac/Unix: `which python`
            - Windows: `where python` 
            - It should be in the *gui_env* directory: 
            - Mac/Unix: `.../env/bin/python`
            - Windows: `...\env\Scripts\python.exe`
        7. Execute the following command: `pip3 install -r requirements.txt`
            > Execute "pip list" to ensure all necessary dependencies are installed.
    - Using Anaconda:
        - TODO

**Fiber Photometry GUI should now be installed and ready to use**

You have the option of utilizing the FiberPhotometry GUI either through a simple python command or using Jupyter Notebook. The instructions for each are below -

Run with Python Script:
If you would like to deploy the server through the terminal *(recommended)*, follow the below instructions:
Navigate to the location of the FiberGui folder/download in your terminal and run the following command -

`panel serve --show FiberGuiScript.py --websocket-max-message-size=104876000 --autoreload`

This command will launch the GUI in a your browser window or tab. \
**Code changes refreshes entire instance**

Run with Jupyter Notebook:
If you would like to utilize Jupyter Notebook to deploy the server, simply open up the notebook (.ipynb) file in a Jupyter environment (i.e. JupyterLab, VSCode, etc.). Then execute each cell individually, *making sure* to let each cell finish execution before continuing to the next. Upon execution of the final cell, a local URL will be produced that navigates to the interface `(e.g. http://localhost:#####)`.
