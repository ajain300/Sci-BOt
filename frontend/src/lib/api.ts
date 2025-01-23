import axios from "axios";
import { API_BASE_URL } from "./utils";

export interface OptimizationConfig {
  objective_function: string;
  constraints: string[];
  bounds: { [key: string]: [number, number] };
  parameters: string[];
}

export interface ConfigGenerationRequest {
  prompt: string;
}

export interface ActiveLearningRequest {
  config: OptimizationConfig;
  data: { [key: string]: number[] };
  n_suggestions: number;
}

export interface ActiveLearningResponse {
  suggestions: { [key: string]: number[] };
  uncertainty: number[];
}

export interface ProcessDataRequest {
  config: OptimizationConfig;
  data: { [key: string]: number[] };
}

export interface ProcessDataResponse {
  insights: string[];
  best_point: { [key: string]: number };
  performance_metrics: { [key: string]: number };
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

export const getSuggestions = async (request: ActiveLearningRequest): Promise<ActiveLearningResponse> => {
  const { data } = await api.post("/optimization/suggest", request);
  return data;
};

export const analyzeData = async (request: ProcessDataRequest): Promise<ProcessDataResponse> => {
  const { data } = await api.post("/optimization/analyze", request);
  return data;
}; 