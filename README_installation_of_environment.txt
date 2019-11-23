##########################################################
Operating system : Windows
Cuda : 8.0
Cudnn : 5.1
##########################################################
download Anaconda from https://www.anaconda.com/download/#windows
Python 3.6 version 

Place the environment.yml file in your pc at any suitable directory. 
Now, open the Anaconda command prompt.
If, the provided environment.yml (located in Spectrum folder) file is in X:\xyz\environment.yml,
then type the following line:

conda-env create -n RR -f X:\xyz\environment.yml

It will create a new environment and install most of the necessary 
libraries

##########################################################
activate the created environment by typing the following line in 
anaconda command prompt:

activate RR

Then install the following libraries manually by writing these lines
in the anaconda command prompt:

pip install --ignore-installed --upgrade tensorflow-gpu==1.4.0
pip install keras==2.1.4
pip install -U pydicom
pip install pynrrd
conda install scikit-image

#####################################
Run the following commands to run our provided python scripts:

activate RR
spyder

Run the codes in the Spyder IDE
########################################

After generating the results you can delete our environment to free
up your pc
deactivate RR
conda env remove -n RR
