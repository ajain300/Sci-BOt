# Sci-BOt: Driving Scientific Optimization using Natural Language

A monorepo containing a Next.js frontend and FastAPI backend for scientific optimization using Bayesian optimization and active learning, enhanced by LLMs.

## Project Structure

```
sci-llm/
├── frontend/                # Next.js frontend application
│   ├── src/                # Source code
│   │   ├── app/           # Next.js app router pages
│   │   ├── components/    # React components
│   │   ├── lib/           # Utilities and API client
│   │   ├── hooks/         # React hooks
│   │   └── __tests__/     # Frontend tests
│   ├── public/            # Static assets
│   └── package.json       # Frontend dependencies
├── backend/               # FastAPI backend application
│   ├── app/              # Main application
│   │   ├── api/          # API routes
│   │   ├── core/         # Core business logic
│   │   ├── schemas/      # Pydantic schemas
│   │   └── utils/        # Utility functions
│   ├── tests/            # Backend tests
│   ├── main.py           # Application entry point
│   └── requirements.txt   # Python dependencies
├── dataset/              # Sample datasets and configurations
│   ├── data_format_example.csv  # Example data format
│   ├── json_example.json        # Example configuration
│   └── test_problems.txt        # Test problem descriptions
├── active_learning/      # Active learning module
├── pyproject.toml        # Python project configuration
└── package.json          # Root dependencies
```

## Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- Poetry (optional) or pip

## Setup

1. Install root dependencies:
   ```bash
   npm install
   ```

2. Set up backend:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up frontend:
   ```bash
   cd frontend
   npm install
   ```

4. Create `.env` files:
   - Backend (.env):
     ```
     OPENAI_API_KEY=your_key_here
     ```
   - Frontend (.env.local):
     ```
     NEXT_PUBLIC_API_URL=http://localhost:8000
     ```

## Development

Run both frontend and backend in development mode:
```bash
npm run dev
```

Or run them separately:
- Frontend only: `npm run frontend`
- Backend only: `npm run backend`

## Features

- Modern Next.js frontend with TypeScript and Tailwind CSS
- FastAPI backend with async support
- Active Learning and Bayesian Optimization
- LLM integration for configuration generation
- Real-time data visualization
- Type-safe API communication

## Testing

- Backend: `cd backend && pytest`
- Frontend: `cd frontend && npm test`

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request