import matplotlib.pyplot as plt
from ModelWrapper import ModelWrapper

print("""\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n

ooo        ooooo   .oooooo.   oooooooooo.   oooooooooooo ooooo             ooooooooooooo oooooooooooo  .oooooo..o ooooooooooooo oooooooooooo ooooooooo.   
`88.       .888'  d8P'  `Y8b  `888'   `Y8b  `888'     `8 `888'             8'   888   `8 `888'     `8 d8P'    `Y8 8'   888   `8 `888'     `8 `888   `Y88. 
 888b     d'888  888      888  888      888  888          888                   888       888         Y88bo.           888       888          888   .d88' 
 8 Y88. .P  888  888      888  888      888  888oooo8     888                   888       888oooo8     `"Y8888o.       888       888oooo8     888ooo88P'  
 8  `888'   888  888      888  888      888  888    "     888                   888       888    "         `"Y88b      888       888    "     888`88b.    
 8    Y     888  `88b    d88'  888     d88'  888       o  888       o           888       888       o oo     .d8P      888       888       o  888  `88b.  
o8o        o888o  `Y8bood8P'  o888bood8P'   o888ooooood8 o888ooooood8          o888o     o888ooooood8 8""88888P'      o888o     o888ooooood8 o888o  o888o 
      
            \n\n\n\n\n""")

TEST_VIDEO_IDS  = ["6", "13"]
TRUE_FOLDER = "./true"
COLUMN_NAMES = {0 : "background", 1 : "supportedrear", 2 : "unsupportedrear", 3 : "grooming"}
OUTPUT_FOLDER = "./output"
SMOOTHING = "gap"
SMOOTHING_WINDOW = 5
CONFUSION_MATRIX_NORMALIZE = True
PREDICTIONS_FOLDER_RADU = "./predictions/RADU"

radu = ModelWrapper(name = "RADU", test_set = TEST_VIDEO_IDS, predictions_folder = PREDICTIONS_FOLDER_RADU, true_folder = TRUE_FOLDER, output_folder = OUTPUT_FOLDER, column_names = COLUMN_NAMES, smoothing = SMOOTHING, smoothing_window = SMOOTHING_WINDOW)
radu.plot_confusion_matrix(normalize = CONFUSION_MATRIX_NORMALIZE)

