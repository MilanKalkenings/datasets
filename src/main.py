from data_handling import LOFFHandler
import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "milankalkenings==0.1.20"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])


# data set: litter on forest floor
loff_handler = LOFFHandler(verbose=1)
loff_handler.handle_data()
