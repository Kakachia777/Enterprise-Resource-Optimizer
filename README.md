# Enterprise Resource Optimizer (ERO)

An advanced enterprise resource optimization system built with FastAPI, featuring ML-powered forecasting, resource optimization, and real-time analytics.

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.2-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Code Style](https://img.shields.io/badge/code%20style-black-black)

## Features

- **Advanced Analytics**
  - ML-powered demand forecasting using ensemble methods
  - Time series analysis with Prophet
  - Real-time resource optimization
  - Network optimization using graph theory

- **Resource Management**
  - Comprehensive resource tracking
  - Location management
  - Transaction history
  - Optimization algorithms

- **Machine Learning Integration**
  - Random Forest, XGBoost, and Prophet models
  - Feature engineering for time series
  - Model versioning and persistence
  - Automated retraining

- **Production-Ready**
  - Containerized deployment
  - Kubernetes configurations
  - CI/CD pipeline
  - Comprehensive monitoring

## Tech Stack

- **Backend**: FastAPI, Python 3.9
- **Database**: PostgreSQL, Redis
- **ML/Analytics**: scikit-learn, XGBoost, Prophet, pandas
- **Optimization**: PuLP, NetworkX, DEAP
- **Monitoring**: Prometheus, Grafana
- **Deployment**: Docker, Kubernetes
- **Testing**: pytest, coverage
- **Documentation**: OpenAPI (Swagger)

## Project Structure

```
supply_chain_api/
├── src/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   ├── models/
│   ├── services/
│   │   ├── analytics.py
│   │   ├── inventory.py
│   │   ├── ml_forecasting.py
│   │   └── optimization.py
│   └── utils/
├── tests/
├── k8s/
├── monitoring/
└── docs/
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/Kakachia777/enterprise-resource-optimizer.git
cd enterprise-resource-optimizer
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

5. Start services with Docker Compose:
```bash
docker-compose up -d
```

6. Run migrations:
```bash
alembic upgrade head
```

## API Documentation

Once running, access the API documentation at:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## Key Endpoints

### Inventory Management
```bash
# Create inventory item
POST /api/v1/inventory/items/

# Get inventory items
GET /api/v1/inventory/items/

# Create transaction
POST /api/v1/inventory/transactions/
```

### Analytics
```bash
# Get demand forecast
GET /api/v1/analytics/demand-forecast/{item_id}

# Get optimization recommendations
GET /api/v1/analytics/optimization/

# Get analytics dashboard
GET /api/v1/analytics/dashboard/
```

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/api/test_inventory.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Run linter
flake8 src/ tests/

# Type checking
mypy src/
```

## Deployment

### Docker
```bash
# Build image
docker build -t supply-chain-api .

# Run container
docker run -p 8000:8000 supply-chain-api
```

### Kubernetes
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/
```

## Monitoring

The system exposes metrics for Prometheus at `/metrics`, including:
- Request latencies
- Database operations
- ML model performance
- Business metrics

Access Grafana dashboards at http://localhost:3000 for:
- System metrics
- Business KPIs
- ML model performance
- API usage statistics

## Security

- API key authentication
- Rate limiting
- Input validation
- SQL injection prevention
- CORS configuration
- Secure password hashing

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Kakachia777 - [@Kakachia777](https://github.com/Kakachia777)

Project Link: [https://github.com/Kakachia777/enterprise-resource-optimizer](https://github.com/Kakachia777/enterprise-resource-optimizer) 