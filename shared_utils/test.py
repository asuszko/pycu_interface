from numpy.ctypeslib import load_library


lib_path = r"C:\Users\Art\Documents\python\imaging_radar\angle_finding\iaa\pycu_interface\cufft_helpers\lib"
lib_name = "cufft"
c_lib = load_library(lib_name+".dll", lib_path)
