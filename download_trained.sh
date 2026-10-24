#!/bin/bash
# Download trained BM3 models from Google Drive
# Models were trained on Amazon datasets (Baby, Sports, Electronics)

echo "BM3 Trained Models Downloader"
echo "=================================="
echo ""
echo "Available trained models:"
echo "1. Baby dataset (385M)"
echo "2. Sports dataset (984M)"  
echo "3. Elec dataset (3.34G)"
echo ""

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown for Google Drive downloads..."
    pip install gdown
fi

# Create models directory
mkdir -p models/trained

echo "Please select a dataset:"
echo "1. Baby"
echo "2. Sports"
echo "3. Elec"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Downloading Baby dataset model..."
        # Google Drive folder ID for Baby
        FOLDER_ID="1OmRY9Z9fC01HjzT1qDuUZEwUBq03oZ5s"
        gdown --folder https://drive.google.com/drive/folders/${FOLDER_ID} -O models/trained/baby/
        echo "Baby model downloaded to models/trained/baby/"
        ;;
    2)
        echo "Downloading Sports dataset model..."
        # Google Drive folder ID for Sports
        FOLDER_ID="12DngCZbMafipy2cpcWEyHz9kmKejOGcu"
        gdown --folder https://drive.google.com/drive/folders/${FOLDER_ID} -O models/trained/sports/
        echo "Sports model downloaded to models/trained/sports/"
        ;;
    3)
        echo "Downloading Elec dataset model..."
        # Google Drive folder ID for Elec
        FOLDER_ID="1QSojeVMGQvGsgNE5GD4T7ZA7ncllWU5N"
        gdown --folder https://drive.google.com/drive/folders/${FOLDER_ID} -O models/trained/elec/
        echo "Elec model downloaded to models/trained/elec/"
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "Download complete!"
echo ""
echo "To test the model, run:"
echo "cd src"
echo "python load_and_test.py -c ../models/trained/<dataset>/*.pth -d <dataset>"
