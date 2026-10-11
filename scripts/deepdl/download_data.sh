echo 'Download data from google drive'
gdown 1177AbZMw4GJFaw4Ntm-ZgqK58pViDyTw -O ./deepdl_data.tar.gz
echo 'Extracting data'
tar -xzvf deepdl_data.tar.gz
echo 'Remove tar file'
rm deepdl_data.tar.gz
