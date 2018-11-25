import os, shutil, datetime

class FileSystem():
    """
    Class Description:
    Class Object to handle the querying of File Directories information regarding the files within.
    Author:
    Jacob Taylor Cassady
    """
    @staticmethod
    def log(data, file_name):
        """
        Function Description:
        Logs the data to the location [file_name]. Note all previous information contained within this file will be deleted.
        Author:
        Jacob Taylor Cassady
        """
        try:
            with open(file_name, "a+") as file:
                file.write(data + "\n")

        except PermissionError:
            print("Permission error when accessing file: " + file_name)

    @staticmethod
    def start_log(data, file_name):
        """
        Function Description:
        Logs the data to the location [file_name]. Note all previous information contained within this file will be deleted.
        Author:
        Jacob Taylor Cassady
        """
        with open(file_name, "w+") as file:
            file.write(data + "\n")

    @staticmethod
    def load_evaluation(file_name):
        print("Loading evaluation for file:", file_name)

        with open(file_name) as evaluation_file:
            data = evaluation_file.readlines()

        loss = float(data[0])
        accuracy = float(data[1])

        print("Evaluation loaded: Loss = {} | Accuracy = {}".format(loss, accuracy))

        return loss, accuracy
