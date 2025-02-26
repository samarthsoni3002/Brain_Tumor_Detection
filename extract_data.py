import zipfile


data_path = "brain_tumor_dataset.zip"
extract_path = "brain_tumor_dataset"


with zipfile.ZipFile(data_path,'r') as zip_extract:
    zip_extract.extractall(extract_path)


print("Data has been extracted successfully!")