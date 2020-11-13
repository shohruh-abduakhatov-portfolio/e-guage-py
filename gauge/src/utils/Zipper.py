import os
import zipfile

"""https://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory"""


# todo finish here. So that each folder of daily captures are .zip'ed

def zip_dir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def main(dir_name, path_to_dir):
    try:
        zipf = zipfile.ZipFile(dir_name + '.zip', 'w', zipfile.ZIP_DEFLATED)
        zip_dir(path_to_dir, zipf)
    finally:
        zipf.close()
