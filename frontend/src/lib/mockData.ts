import type { OptimizationConfig } from './api';


// Mock optimization config for testing UI without making LLM calls
export const getMockConfig = (): OptimizationConfig => {
    return {
      features: [
        {
          name: "monomer_composition",
          type: "composition",
          columns: {
            parts: [
              "PEGDA_concentration",
              "IBOA_concentration",
              "IDA_concentration"
            ],
            range: {
              "PEGDA_concentration": [0.0, 100.0],
              "IBOA_concentration": [0.0, 50.0],
              "IDA_concentration": [0.0, 100.0]
            }
          },
          scaling: "lin"
        },
        {
          name: "curing_time",
          type: "continuous",
          min: 0.0,
          max: 240.0,
          scaling: "lin"
        },
        {
          name: "experiment_type",
          type: "discrete",
          categories: ["simulation", "experiment"]
        }
      ],
      objectives: [
        {
          name: "Young's_modulus",
          direction: "maximize",
          weight: 1.0
        },
        {
          name: "Tg",
          direction: "maximize",
          weight: 1.0
        }
      ],
      acquisition_function: "best_score",
      constraints: [
        "IBOA_concentration <= 50"
      ]
    };
  };

// Define a type for the suggestions to avoid issues
export type Suggestion = {
  [key: string]: number | string;
};

// Mock suggestions data for testing
export const getMockSuggestions = (numSuggestions: number = 3) => {
  const suggestions: Suggestion[] = [];
  const expectedImprovements: number[] = [];
  
  for (let i = 0; i < numSuggestions; i++) {
    suggestions.push({
      temperature: 75 + Math.random() * 20,
      pressure: 5 + Math.random() * 4,
      catalyst_type: ["a", "b", "c"][Math.floor(Math.random() * 3)],
      monomer_a: 0.3 + Math.random() * 0.2,
      monomer_b: 0.3 + Math.random() * 0.2,
      monomer_c: 0.3 + Math.random() * 0.2,
    });
    
    expectedImprovements.push(0.1 + Math.random() * 0.3);
  }
  
  return {
    suggestions,
    expected_improvements: expectedImprovements
  };
};