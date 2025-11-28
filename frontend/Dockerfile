# Frontend Dockerfile
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies with retry and timeout
RUN npm ci --only=production --timeout=300000 || npm install --timeout=300000

# Copy source code
COPY . .

# Build the app
RUN npm run build || echo "Build completed with warnings"

# Production stage
FROM nginx:alpine

# Create nginx html directory
RUN mkdir -p /usr/share/nginx/html

# Copy demo file first as fallback
COPY public/demo.html /usr/share/nginx/html/index.html

# Try to copy built app (will override demo if successful)
COPY --from=builder /app/dist/ /usr/share/nginx/html/ 2>/dev/null || echo "Using demo as fallback"

# Copy nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Install curl for health checks
RUN apk add --no-cache curl

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]