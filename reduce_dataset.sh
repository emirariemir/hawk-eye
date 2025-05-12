#!/bin/bash

# Shell script to reduce COCO dataset to 10,000 images
echo "🔧 Creating reduced dataset directory..."
mkdir -p archive/train_10k/images

echo "📁 Copying first 10,000 images..."
ls archive/train/images | head -n 10000 | xargs -I{} cp archive/train/images/{} archive/train_10k/images/

echo "✂️  Filtering annotations.json..."
python3 reduce_annotations.py

echo "✅ Done! Your reduced dataset is in archive/train_10k"
