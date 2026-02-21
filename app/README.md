## About
This is the user application. It consists of three Docker containers:
- **Frontend**: Powered by Angular (TypeScript).
- **Backend**: Powered by Flask (Python).
- **Ollama**: Local LLM service for generating prompts.

## Usage
To start the entire application (Frontend, Backend, and Ollama), run:
```
docker compose up --build
```

This will start:
- Frontend at `http://localhost:4200/`
- Backend at `http://localhost:5001/`
- Ollama at `http://localhost:11434/`

The `llama3.2` model is automatically pulled into the Ollama container.

## Frontend Development
For faster frontend development, building a Docker container is not required. You can instead run:
```
cd frontend
ng serve
```
However, this requires installing node.js, npm, angular, and other packages.

## Backend Development
Developing the backend requires rebuilding and restarting the backend docker.

## Run frontend locally
```
cd frontend
npm install
npm start

docker run -it --rm -d -p 8080:80 --name web nginx

cd ../backend
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
python3 app.py
```