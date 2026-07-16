source venv/bin/activate 
echo "✅ venv activated"

python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
echo "✅ ipykernel installed"

