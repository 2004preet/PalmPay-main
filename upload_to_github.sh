#!/bin/bash
# PalmPay - Upload to GitHub Script

cd "$(dirname "$0")"

echo ""
echo "=========================================="
echo "  🚀 PalmPay - Upload to GitHub"
echo "=========================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed!"
    echo "Install it from: https://git-scm.com/downloads"
    exit 1
fi

echo "Step 1: Checking git status..."
if [ -d .git ]; then
    echo "✅ Git repository already initialized"
    git status --short
else
    echo "📦 Initializing git repository..."
    git init
fi

echo ""
echo "Step 2: Adding files..."
git add .

echo ""
echo "Step 3: Checking what will be committed..."
git status --short

echo ""
read -p "Do you want to commit these changes? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter commit message (or press Enter for default): " commit_msg
    if [ -z "$commit_msg" ]; then
        commit_msg="Initial commit: PalmPay - Palm recognition payment system"
    fi
    
    echo ""
    echo "Step 4: Creating commit..."
    git commit -m "$commit_msg"
    
    echo ""
    echo "=========================================="
    echo "✅ Files committed successfully!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Create a repository on GitHub:"
    echo "   Go to: https://github.com/new"
    echo "   Create a new repository (don't initialize with files)"
    echo ""
    echo "2. Add remote and push:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/PalmPay.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    echo "Or run these commands (replace YOUR_USERNAME):"
    echo ""
    echo "   git remote add origin https://github.com/YOUR_USERNAME/PalmPay.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
else
    echo "Commit cancelled."
fi

echo ""

