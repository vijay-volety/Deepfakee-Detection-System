# Production Deployment Guide

This guide covers deploying the DeepFake Detection System to production environments.

## Cloud Deployment Options

### AWS Deployment

#### Prerequisites
- AWS account with appropriate permissions
- AWS CLI configured
- Docker installed locally

#### GPU Instance Setup (Recommended)

1. **Launch GPU Instance**
   ```bash
   # Launch p3.2xlarge or g4dn.xlarge instance
   aws ec2 run-instances \
     --image-id ami-0c02fb55956c7d316 \
     --instance-type g4dn.xlarge \
     --key-name your-key-name \
     --security-group-ids sg-xxxxxxxxx \
     --subnet-id subnet-xxxxxxxxx
   ```

2. **Install NVIDIA Drivers**
   ```bash
   sudo apt update
   sudo apt install -y nvidia-driver-470
   sudo reboot
   ```

3. **Install Docker and NVIDIA Container Toolkit**
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER

   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt update
   sudo apt install -y nvidia-docker2
   sudo systemctl restart docker
   ```

4. **Deploy Application**
   ```bash
   git clone <repository-url>
   cd deepfake
   cp .env.example .env
   # Edit .env with production values
   docker-compose up -d --build
   ```

#### Load Balancer Setup

```bash
# Create Application Load Balancer
aws elbv2 create-load-balancer \
  --name deepfake-alb \
  --subnets subnet-xxxxxxxxx subnet-yyyyyyyyy \
  --security-groups sg-xxxxxxxxx
```

### Google Cloud Platform (GCP)

#### Setup GPU VM

```bash
# Create GPU instance
gcloud compute instances create deepfake-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE \
  --restart-on-failure
```

#### Install Dependencies

```bash
# SSH into instance
gcloud compute ssh deepfake-gpu --zone=us-central1-a

# Install CUDA and drivers
sudo apt update
sudo apt install -y nvidia-driver-470
curl https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```

### Azure Deployment

#### Create GPU VM

```bash
# Create resource group
az group create --name deepfake-rg --location eastus

# Create GPU VM
az vm create \
  --resource-group deepfake-rg \
  --name deepfake-gpu \
  --image UbuntuLTS \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys
```

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster with GPU nodes
- kubectl configured
- NVIDIA GPU Operator installed

### Deployment Manifests

#### Namespace
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: deepfake
```

#### ConfigMap
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: deepfake-config
  namespace: deepfake
data:
  DATABASE_URL: "postgresql://user:pass@postgres:5432/deepfake"
  REDIS_URL: "redis://redis:6379"
  JWT_SECRET_KEY: "your-production-secret"
```

#### Inference Service Deployment
```yaml
# inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
  namespace: deepfake
spec:
  replicas: 2
  selector:
    matchLabels:
      app: inference-service
  template:
    metadata:
      labels:
        app: inference-service
    spec:
      containers:
      - name: inference
        image: deepfake-inference:latest
        ports:
        - containerPort: 8001
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"
          requests:
            nvidia.com/gpu: 1
            memory: "4Gi"
            cpu: "1"
        envFrom:
        - configMapRef:
            name: deepfake-config
```

#### Apply Deployments
```bash
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f inference-deployment.yaml
kubectl apply -f backend-deployment.yaml
kubectl apply -f frontend-deployment.yaml
kubectl apply -f services.yaml
```

## Database Setup

### PostgreSQL (Recommended for Production)

#### Docker Compose Override
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: deepfake
      POSTGRES_USER: deepfake_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  backend:
    environment:
      DATABASE_URL: postgresql+asyncpg://deepfake_user:secure_password@postgres:5432/deepfake

volumes:
  postgres_data:
```

#### Managed Database Services

**AWS RDS:**
```bash
aws rds create-db-instance \
  --db-instance-identifier deepfake-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username deepfake_user \
  --master-user-password secure_password \
  --allocated-storage 20
```

## SSL/TLS Configuration

### Reverse Proxy with Nginx

```nginx
# nginx.prod.conf
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/ssl/certs/yourdomain.com.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.com.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    client_max_body_size 100M;

    location / {
        proxy_pass http://frontend:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Let's Encrypt with Certbot

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Monitoring and Logging

### Prometheus and Grafana

```yaml
# monitoring/docker-compose.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  grafana_data:
```

### ELK Stack for Logging

```yaml
# logging/docker-compose.yml
version: '3.8'
services:
  elasticsearch:
    image: elasticsearch:7.14.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  kibana:
    image: kibana:7.14.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

  logstash:
    image: logstash:7.14.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
```

## Backup and Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_NAME="deepfake"

# PostgreSQL backup
pg_dump -h postgres -U deepfake_user $DB_NAME | gzip > $BACKUP_DIR/db_backup_$DATE.sql.gz

# Upload to S3
aws s3 cp $BACKUP_DIR/db_backup_$DATE.sql.gz s3://your-backup-bucket/database/

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +7 -delete
```

### Model Backup

```bash
# Backup trained models
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
MODEL_DIR="/app/models/checkpoints"
BACKUP_DIR="/backups/models"

tar -czf $BACKUP_DIR/models_$DATE.tar.gz $MODEL_DIR
aws s3 cp $BACKUP_DIR/models_$DATE.tar.gz s3://your-backup-bucket/models/
```

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  inference:
    deploy:
      replicas: 3
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  backend:
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G
          cpus: '1'
```

### Auto-scaling with Kubernetes

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
  namespace: deepfake
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Security Best Practices

### Network Security

1. **VPC Configuration**
   - Private subnets for backend services
   - Public subnet only for load balancer
   - Security groups with minimal required access

2. **Firewall Rules**
   ```bash
   # Only allow necessary ports
   sudo ufw allow 22    # SSH
   sudo ufw allow 80    # HTTP
   sudo ufw allow 443   # HTTPS
   sudo ufw enable
   ```

### Application Security

1. **Environment Variables**
   - Use secrets management (AWS Secrets Manager, Azure Key Vault)
   - Rotate secrets regularly
   - Never commit secrets to version control

2. **Rate Limiting**
   - Implement at reverse proxy level
   - Configure application-level rate limiting
   - Monitor for abuse patterns

### Data Security

1. **Encryption**
   - Encrypt data at rest
   - Use HTTPS for all communications
   - Encrypt sensitive model data

2. **Access Control**
   - Implement proper authentication
   - Use role-based access control
   - Regular security audits

## Performance Optimization

### GPU Optimization

```yaml
# Optimize GPU memory usage
services:
  inference:
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_MEMORY_FRACTION=0.8
```

### Caching Strategy

```yaml
# Redis configuration for caching
services:
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

## Maintenance

### Regular Updates

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker images
docker-compose pull
docker-compose up -d --force-recreate

# Cleanup unused images
docker system prune -af
```

### Health Checks

```bash
# Automated health check script
#!/bin/bash
curl -f http://localhost:8000/api/v1/health || exit 1
curl -f http://localhost:8001/health || exit 1
curl -f http://localhost:3000 || exit 1
```

This deployment guide provides comprehensive instructions for production deployment across different cloud platforms and container orchestration systems.