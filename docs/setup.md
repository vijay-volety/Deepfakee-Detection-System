# DeepFake Detection System - Setup Guide

This guide will help you set up and run the DeepFake Detection System locally or deploy it to production.

## Prerequisites

- Docker and Docker Compose
- (Optional) NVIDIA GPU with CUDA support for better performance
- (Optional) Node.js and Python for development

## Quick Start (Docker)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd deepfake
```

### 2. Environment Setup

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit the `.env` file with your configurations:

```env
# Security
JWT_SECRET_KEY=your-super-secret-jwt-key-here
DATABASE_URL=sqlite+aiosqlite:///./deepfake.db

# Redis
REDIS_URL=redis://redis:6379

# API URLs
REACT_APP_API_URL=http://localhost:8000
REACT_APP_INFERENCE_URL=http://localhost:8001

# Model Configuration
MODEL_PATH=/models
CUDA_VISIBLE_DEVICES=0
```

### 3. Start the Services

For GPU-enabled deployment:
```bash
docker-compose up --build
```

For CPU-only deployment:
```bash
docker-compose -f docker-compose.cpu.yml up --build
```

### 4. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Inference Service: http://localhost:8001

## Development Setup

### Backend Development

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

### Inference Service Development

```bash
cd inference
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8001
```

### Model Training

```bash
cd models
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download datasets (see Data Preparation section)
python scripts/preprocess.py --config configs/train_config.yaml

# Train model
python scripts/train.py --config configs/train_config.yaml

# Evaluate model
python scripts/evaluate.py --model checkpoints/best_model.pth --config configs/train_config.yaml
```

## Data Preparation

### Dataset Download

1. **DFDC Dataset**
   ```bash
   # Register at https://dfdc.ai/ and download
   mkdir -p models/data/DFDC
   # Extract dataset to models/data/DFDC/
   ```

2. **FaceForensics++ Dataset**
   ```bash
   # Register at https://github.com/ondyari/FaceForensics
   mkdir -p models/data/FaceForensics++
   # Follow their download instructions
   ```

3. **Celeb-DF Dataset**
   ```bash
   # Register at https://github.com/yuezunli/celeb-deepfakeforensics
   mkdir -p models/data/Celeb-DF-v2
   # Download as per their instructions
   ```

### Dataset Structure

```
models/data/
├── DFDC/
│   ├── train/
│   ├── val/
│   └── test/
├── FaceForensics++/
│   ├── train/
│   ├── val/
│   └── test/
└── Celeb-DF-v2/
    ├── train/
    ├── val/
    └── test/
```

## Configuration

### Model Configuration

Edit `models/configs/train_config.yaml`:

```yaml
model:
  resnet_type: "resnet50"
  lstm_hidden_size: 512
  lstm_num_layers: 2
  dropout_rate: 0.5

training:
  num_epochs: 100
  batch_size: 8
  learning_rate: 0.0001

data:
  sequence_length: 16
  datasets: ["DFDC", "FaceForensics++", "Celeb-DF"]
```

### Backend Configuration

Key environment variables:

- `JWT_SECRET_KEY`: Secret key for JWT tokens
- `DATABASE_URL`: Database connection URL
- `REDIS_URL`: Redis connection URL
- `MAX_FILE_SIZE`: Maximum upload file size (bytes)
- `ALLOWED_FILE_TYPES`: Comma-separated list of allowed extensions

### Frontend Configuration

Environment variables for React:

- `REACT_APP_API_URL`: Backend API URL
- `REACT_APP_INFERENCE_URL`: Inference service URL

## Troubleshooting

### Common Issues

1. **GPU not detected**
   ```bash
   # Check NVIDIA Docker runtime
   docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
   ```

2. **Out of memory errors**
   - Reduce batch size in training config
   - Use CPU-only deployment for testing

3. **Port conflicts**
   - Change ports in docker-compose.yml
   - Update environment variables accordingly

4. **Database issues**
   ```bash
   # Reset database
   docker-compose down -v
   docker-compose up --build
   ```

### Performance Optimization

1. **GPU Memory**
   - Monitor with `nvidia-smi`
   - Adjust batch sizes based on available memory

2. **Storage**
   - Use SSD for better I/O performance
   - Consider using external storage for large datasets

3. **Network**
   - Use CDN for static assets in production
   - Enable compression in reverse proxy

## Production Deployment

See [deployment.md](deployment.md) for detailed production deployment instructions.

## API Usage

See [api.md](api.md) for detailed API documentation and examples.

## Model Training

See [training.md](training.md) for comprehensive model training guide.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs: `docker-compose logs <service-name>`
3. Open an issue on GitHub with detailed error information