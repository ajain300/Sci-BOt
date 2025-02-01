# Sci-Opt-LLM: Scientific Optimization with LLM Frontend

A monorepo containing a Next.js frontend and FastAPI backend for scientific optimization using Bayesian optimization and active learning, powered by LLMs.

## Project Structure

```
sci-opt-llm/
├── frontend/          # Next.js frontend application
├── backend/           # FastAPI backend application
│   ├── app/
│   │   ├── api/      # API routes
│   │   ├── core/     # Core business logic
│   │   ├── models/   # Database models
│   │   ├── schemas/  # Pydantic schemas
│   │   └── utils/    # Utility functions
│   └── tests/        # Backend tests
└── dataset/          # Sample datasets and configurations
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
