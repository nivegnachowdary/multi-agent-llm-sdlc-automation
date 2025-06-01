import zipfile
import os

def zip_output():
    zip_filename = "output_bundle.zip"
    zip_path = f"output/{zip_filename}"

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for folder, _, files in os.walk("output"):
            for file in files:
                if file != zip_filename:
                    zipf.write(os.path.join(folder, file), arcname=file)

    return zip_path