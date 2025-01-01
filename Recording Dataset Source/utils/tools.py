import os
import argparse

class Tools():
    def __init__(self):
        pass
    
    def getAbsPath(filePath):
        abs_path = os.path.abspath(filePath)
        print(abs_path)
        return abs_path

    def convert_slashes(directory_path):
        return directory_path.replace("\\", "/")

    def parse_arguments() -> argparse.Namespace: # For Camera
        parser=argparse.ArgumentParser(description="YOLOv8 Live")
        parser.add_argument(
            "--webcam-resolution",
            default=[1280,720], #default must be 1280, 720
            nargs=2,
            type=int
        )
        args = parser.parse_args()
        return args
    
    def addAction(number_of_Sequences):
        noOfSequences = number_of_Sequences
        
        DATA_PATH = os.path.join('THESIS_FILES', 'HumanPose_DATA') 
        actionsList = []
        #actionInput = folderNameInput.get(1.0, "end-1c")

        
        # if actionInput == '':
        #     messagebox.showinfo("Information", "Please enter action name!")
        # else:
        #     actionsList.append(actionInput)
        #     for action in actionsList:
        #         for sequence in range(1, noOfSequences+1):
        #             try: 
        #                 os.makedirs(os.path.join(DATA_PATH, action,
        #                                         str(sequence)))
        #             except FileExistsError:
        #                 flagExist = True
        #                 pass
        #     if flagExist:
        #         messagebox.showerror(title="Already Existed!", message="The name of action you entered already exists!")
        #     folderNameInput.delete("1.0", "end")

