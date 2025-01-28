import axios from "axios";
import { API_BASE_URL } from "./utils";

export interface OptimizationConfig {
  parameters: {
    [key: string]: {
      min: number | "n/a";
      max: number | "n/a";
      type: "continuous" | "discrete" | "derived";
      derived_from?: string;
    };
  };
  objective: string;
  objective_variable: string;
  constraints?: string[];
}

export interface ConfigGenerationRequest {
  prompt: string;
}

export interface ActiveLearningRequest {
  config: OptimizationConfig;
  data: Array<{
    parameters: { [key: string]: number };
    objective_value: number;
  }>;
  n_suggestions: number;
}

export interface ActiveLearningResponse {
  suggestions: Array<{ [key: string]: number }>;
  expected_improvements: number[];
}

export interface ProcessDataRequest {
  config: OptimizationConfig;
  data: { [key: string]: number[] };
}

export interface ProcessDataResponse {
  statistics: { [key: string]: number };
  best_point: {
    parameters: { [key: string]: number };
    objective_value: number | null;
  };
  parameter_importance: { [key: string]: number };
  data: { [key: string]: number[] };
}

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const generateConfig = async (prompt: string): Promise<OptimizationConfig> => {
  const { data } = await api.post("/optimization/config", { prompt });
  return data;
};

export const getCSVTemplate = async (config: OptimizationConfig): Promise<string> => {
  const response = await api.post("/optimization/template", config, {
    responseType: 'text'
  });
  return response.data;
};

export const getSuggestions = async (request: ActiveLearningRequest): Promise<ActiveLearningResponse> => {
  const { data } = await api.post("/optimization/suggest", request);
  return data;
};

export const analyzeData = async (request: ProcessDataRequest): Promise<ProcessDataResponse> => {
  const { data } = await api.post("/optimization/analyze", request);
  return data;
}; 