set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

conda install -c pytorch faiss-cpu -y
sudo apt update
sudo apt install default-jdk
conda install -c conda-forge openjdk=21 maven -y

git clone https://github.com/ZihanWang314/webshop-minimal.git

# webshop installation, model loading
pip install -e webshop-minimal/ --no-dependencies
pip install -U "numpy==2.3.2" "scipy==1.16.1"
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

conda install conda-forge::gdown -y
mkdir -p webshop-minimal/webshop_minimal/data/full
cd webshop-minimal/webshop_minimal/data/full
gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB # items_shuffle
gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi # items_ins_v2
cd ../../../..

echo -e "${GREEN}Installation completed successfully!${NC}"
