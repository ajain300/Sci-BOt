import axios from "axios";
import { API_BASE_URL } from "./utils";

type OptimizationDirection = "maximize" | "minimize" | "target" | "range";
type AcquisitionFunction = "expected_improvement" | "diversity_uncertainty" | "best_score" | "combined_single_ei";

interface BaseFeature {
  name: string;
  type: string;
  scaling?: string;
}

interface ContinuousFeature extends BaseFeature {
  type: "continuous";
  min: number;
  max: number;
}

interface DiscreteFeature extends BaseFeature {
  type: "discrete";
  categories: string[];
}

interface CompositionFeature extends BaseFeature {
  type: "composition";
  columns: {
    parts: string[];
    range: {
      [key: string]: [number, number];
    };
  };
}

export interface ObjectiveConfig {
  name: string;
  weight: number;
  direction: OptimizationDirection;
  target_value?: number;
  range_min?: number;
  range_max?: number;
  unit?: string;
  scaling?: string;
}

export interface OptimizationConfig {
  features: Array<ContinuousFeature | DiscreteFeature | CompositionFeature>;
  objectives: ObjectiveConfig[];
  acquisition_function: AcquisitionFunction;
  constraints?: string[];
}

export interface ConfigGenerationRequest {
  prompt: string;
}

export interface ActiveLearningRequest {
  config: OptimizationConfig;
  data: Array<{
    parameters: { [key: string]: number | string };
    objective_values: { [key: string]: number };
  }>;
  n_suggestions: number;
}

export interface SuggestionResponse {
  rank: number;
  suggestion: { [key: string]: number | string };
  predictions: Array<{ [key: string]: number | string }>;
  reason: string;
}

export interface ActiveLearningResponse {
  suggestions: SuggestionResponse[];
}

export interface ProcessDataRequest {
  config: OptimizationConfig;
  data: Array<{
    parameters: { [key: string]: number | string };
    objective_values: { [key: string]: number };
  }>;
}

export interface ProcessDataResponse {
  statistics: { [key: string]: any };
  best_point: {
    parameters: { [key: string]: number | string };
    objective_values: { [key: string]: number };
  };
  parameter_importance: { [key: string]: number };
}

export const api = axios.create({
  baseURL: API_BASE_URL.endsWith('/') ? API_BASE_URL.slice(0, -1) : API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.data) {
      return Promise.reject(error.response.data);
    }
    return Promise.reject(error);
  }
);

export const generateConfig = async (prompt: string): Promise<OptimizationConfig> => {
  const { data } = await api.post("/optimization/config", { prompt });
  return data;
};

export const getCSVTemplate = async (config: OptimizationConfig): Promise<string> => {
  try {
    const response = await api.post("/optimization/template", config, {
      responseType: 'text',
      headers: {
        'Accept': 'text/csv'
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching CSV template:', error);
    throw new Error('Failed to generate CSV template');
  }
};

export const getSuggestions = async (request: ActiveLearningRequest): Promise<ActiveLearningResponse> => {
  const { data } = await api.post("/optimization/suggest", request);
  return data;
};

export const analyzeData = async (request: ProcessDataRequest): Promise<ProcessDataResponse> => {
  const { data } = await api.post("/optimization/analyze", request);
  return data;
}; 