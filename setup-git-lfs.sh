#!/bin/bash
set -e

# Check if Git LFS is installed
if ! command -v git-lfs &>/dev/null; then
    echo "Git LFS is not installed. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install git-lfs
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update
        sudo apt-get install -y git-lfs
    else
        echo "Unsupported OS. Please install Git LFS manually: https://git-lfs.github.com/"
        exit 1
    fi
fi

# Initialize Git LFS
git lfs install

# Track large binary files
git lfs track "*.png"
git lfs track "*.jpg"
git lfs track "*.jpeg"
git lfs track "*.gif"
git lfs track "*.webp"
git lfs track "*.pdf"
git lfs track "*.zip"
git lfs track "*.tar.gz"

# Add .gitattributes to git
git add .gitattributes

echo "Git LFS setup complete! You can now commit large files."
echo "Run 'git lfs migrate import --include=\"*.png,*.jpg,*.jpeg,*.gif,*.webp,*.pdf,*.zip,*.tar.gz\" --everything' to migrate existing files."
