"use client";

import { useState } from "react";
import { toast } from "react-hot-toast";
import { getSuggestions } from "@/lib/api";
import type { OptimizationConfig } from "@/lib/api";
import FileUpload from "./FileUpload";

interface Props {
  config: OptimizationConfig;
}

export default function ActiveLearning({ config }: Props) {
  const [suggestions, setSuggestions] = useState<Array<{ [key: string]: number }> | null>(null);
  const [expectedImprovements, setExpectedImprovements] = useState<number[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [numSuggestions, setNumSuggestions] = useState(3);
  const [historicalData, setHistoricalData] = useState<Array<{
    parameters: { [key: string]: number };
    objective_value: number;
  }> | null>(null);

  // Get list of non-derived parameters
  const nonDerivedParams = Object.entries(config.parameters)
    .filter(([_, param]) => param.type !== "derived")
    .map(([name]) => name);

  const handleDataLoaded = async (data: { [key: string]: number[] }) => {
    try {
      console.log('Raw data from file:', data);
      console.log('Non-derived parameters:', nonDerivedParams);
      console.log('Objective variable:', config.objective_variable);

      // Transform data into the format backend expects
      const transformedData = data[config.objective_variable].map((_, index) => {
        // Only include non-derived parameters in parameters object
        const parameters: { [key: string]: number } = {};
        nonDerivedParams.forEach(paramName => {
          if (data[paramName] && data[paramName][index] !== undefined) {
            parameters[paramName] = data[paramName][index];
          }
        });

        const dataPoint = {
          parameters,
          objective_value: data[config.objective_variable][index]
        };
        console.log(`Transformed data point ${index}:`, dataPoint);
        return dataPoint;
      });

      // Validate transformed data
      if (transformedData.length === 0) {
        throw new Error("No valid data points found in file");
      }

      // Validate each data point has all required parameters
      const missingParams = transformedData.some(point => 
        nonDerivedParams.some(param => point.parameters[param] === undefined)
      );
      if (missingParams) {
        throw new Error("Some data points are missing required parameters");
      }

      console.log('Final transformed data:', transformedData);
      setHistoricalData(transformedData);
      toast.success(`Loaded ${transformedData.length} data points successfully!`);
    } catch (error) {
      console.error('Error processing data:', error);
      toast.error(error instanceof Error ? error.message : "Error processing data");
    }
  };

  const generateSuggestions = async () => {
    try {
      setIsLoading(true);
      
      // Ensure we're only sending non-derived parameters in the config
      const cleanConfig = {
        ...config,
        parameters: Object.fromEntries(
          Object.entries(config.parameters)
            .filter(([_, param]) => param.type !== "derived")
        )
      };
      
      const request = {
        config: cleanConfig,
        data: historicalData || [], // Use empty array if no historical data
        n_suggestions: numSuggestions
      };

      console.log('Clean config:', cleanConfig);
      console.log('Historical data being sent:', historicalData);
      console.log('Full request:', JSON.stringify(request, null, 2));
      
      const response = await getSuggestions(request);
      console.log('API Response:', response);
      
      setSuggestions(response.suggestions);
      setExpectedImprovements(response.expected_improvements);
      toast.success("Generated new suggestions!");
    } catch (error) {
      console.error('API Error:', error);
      const errorMessage = error instanceof Error ? error.message : "Failed to generate suggestions";
      if (error instanceof Error && 'response' in error) {
        // @ts-ignore
        console.error('API Error Details:', error.response?.data);
        // @ts-ignore
        toast.error(`Error: ${error.response?.data?.detail || errorMessage}`);
      } else {
        toast.error(errorMessage);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="card p-6">
      <h2 className="text-2xl font-semibold mb-4">Active Learning</h2>
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-medium mb-2">Historical Data (Optional)</h3>
          <FileUpload onDataLoaded={handleDataLoaded} />
          {historicalData && (
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Loaded {historicalData.length} data points
            </p>
          )}
        </div>

        <div>
          <h3 className="text-lg font-medium mb-2">Generate Suggestions</h3>
          <div className="flex items-end gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Number of Suggestions
              </label>
              <input
                type="number"
                min={1}
                max={10}
                value={numSuggestions}
                onChange={(e) => setNumSuggestions(Math.max(1, Math.min(10, parseInt(e.target.value) || 1)))}
                className="input-base w-24 p-2 rounded-md"
              />
            </div>
            <button 
              onClick={generateSuggestions}
              disabled={isLoading}
              className="btn-primary"
            >
              Generate
            </button>
          </div>
        </div>

        {isLoading && (
          <div className="text-center py-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">Generating suggestions...</p>
          </div>
        )}

        {suggestions && (
          <div>
            <h3 className="text-lg font-medium mb-2">Suggested Experiments</h3>
            <div className="space-y-4">
              {suggestions.map((suggestion, suggestionIndex) => (
                <div key={suggestionIndex} className="bg-zinc-50 dark:bg-zinc-800 p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Suggestion {suggestionIndex + 1}</h4>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                    {Object.entries(suggestion).map(([param, value]) => (
                      <div key={param} className="bg-white dark:bg-zinc-900 p-3 rounded-md">
                        <p className="text-sm text-zinc-600 dark:text-zinc-400">{param}</p>
                        <p className="font-mono">{value.toFixed(4)}</p>
                      </div>
                    ))}
                    {expectedImprovements && (
                      <div className="bg-white dark:bg-zinc-900 p-3 rounded-md">
                        <p className="text-sm text-zinc-600 dark:text-zinc-400">Expected Improvement</p>
                        <p className="font-mono">{expectedImprovements[suggestionIndex].toFixed(4)}</p>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 