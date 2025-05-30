/// <reference types="@testing-library/jest-dom" />
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Home from '../app/page';

// Mock react-hook-form
jest.mock('react-hook-form', () => ({
  useForm: () => ({
    register: () => ({}),
    handleSubmit: (fn: any) => (e: any) => e.preventDefault(),
    formState: { errors: {} }
  })
}));

// Mock the API
jest.mock('@/lib/api', () => ({
  generateConfig: jest.fn()
}));

// Mock toast
jest.mock('react-hot-toast', () => ({
  toast: { error: jest.fn(), success: jest.fn() }
}));

describe('Home', () => {
  const mockBackendResponse = {
    features: [
      {
        name: "temperature",
        type: "continuous",
        min: 20.0,
        max: 100.0,
        scaling: "lin"
      },
      {
        name: "material_type",
        type: "discrete",
        categories: ["type_a", "type_b", "type_c"]
      },
      {
        name: "composition",
        type: "composition",
        columns: {
          parts: ["component_a", "component_b"],
          range: {
            component_a: [0, 50],
            component_b: [50, 100]
          }
        }
      }
    ],
    objectives: [
      {
        name: "efficiency",
        direction: "maximize",
        weight: 1.0
      },
      {
        name: "cost",
        direction: "minimize",
        weight: 0.8
      }
    ],
    acquisition_function: "diversity_uncertainty",
    constraints: ["temperature <= 100"]
  };

  it('renders the form', () => {
    render(<Home />);
    expect(screen.getByText('Scientific Optimization')).toBeInTheDocument();
  });

  it('handles configuration from backend', async () => {
    const generateConfig = jest.requireMock('@/lib/api').generateConfig;
    generateConfig.mockResolvedValue(mockBackendResponse);

    render(<Home />);

    // Submit the form
    const input = screen.getByPlaceholderText(/optimize/i);
    const submitButton = screen.getByRole('button', { name: 'Generate Configuration' });
    
    fireEvent.change(input, { target: { value: 'test optimization prompt' } });
    fireEvent.click(submitButton);

    // Wait for and verify the configuration display
    await waitFor(() => {
      // Continuous feature
      expect(screen.getByText('temperature')).toBeInTheDocument();
      expect(screen.getByText('Range: [20, 100]')).toBeInTheDocument();

      // Discrete feature
      expect(screen.getByText('material_type')).toBeInTheDocument();
      expect(screen.getByText('type_a')).toBeInTheDocument();

      // Composition feature
      expect(screen.getByText('composition')).toBeInTheDocument();
      expect(screen.getByText('component_a')).toBeInTheDocument();

      // Objectives
      expect(screen.getByText('efficiency')).toBeInTheDocument();
      expect(screen.getByText('maximize')).toBeInTheDocument();

      // Constraints
      expect(screen.getByText('temperature <= 100')).toBeInTheDocument();
    });
  });
}); 