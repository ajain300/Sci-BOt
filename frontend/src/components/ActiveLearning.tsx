"use client";

import { useState, useEffect } from "react";
import { toast } from "react-hot-toast";
import { getSuggestions } from "@/lib/api";
import type { OptimizationConfig, ActiveLearningRequest, SuggestionResponse } from "@/lib/api";
import FileUpload from "./FileUpload";

// Helper function to format parameter names for display
const formatParamName = (name: string): string => {
  // Convert snake_case to Title Case (e.g., "monomer_composition" -> "Monomer Composition")
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

interface Props {
  config: OptimizationConfig;
}

interface HistoricalDataPoint {
  parameters: { [key: string]: number | string };  // Allow both numbers and strings
  objective_values: { [key: string]: number };
}

// Define a type for the suggestions that matches the API response
type Suggestion = {
  rank: number;
  suggestion: { [key: string]: number | string };
  predictions: Array<{ [key: string]: number | string }>;
  reason: string;
};

// Helper function to standardize keys
const getStandardKey = (key: string): string => key.toLowerCase().replace(/[^a-z0-9]/g, '');

export default function ActiveLearning({ config: propsConfig }: Props) {
  const [suggestions, setSuggestions] = useState<Suggestion[] | null>(null);
  const [expectedImprovements, setExpectedImprovements] = useState<number[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("Generating suggestions...");
  const [numSuggestions, setNumSuggestions] = useState(3);
  const [historicalData, setHistoricalData] = useState<HistoricalDataPoint[] | null>(null);
  const [config, setConfig] = useState<OptimizationConfig>(propsConfig);

  // Rotate loading messages
  useEffect(() => {
    if (!isLoading) return;
    
    const messages = [
      "Training backend model...",
      "Enumerating design space...",
      "Evaluating candidates...",
      "Generating suggestions..."
    ];
    let index = 0;
    
    const interval = setInterval(() => {
      index = (index + 1) % messages.length;
      setLoadingMessage(messages[index]);
    }, 3000);
    
    return () => clearInterval(interval);
  }, [isLoading]);

  // Update config when props change
  useEffect(() => {
    setConfig(propsConfig);
  }, [propsConfig]);

  // Update historical data when config changes
  useEffect(() => {
    if (historicalData && config) {
      const updatedData = historicalData.map(point => {
        const newPoint = {
          parameters: {} as { [key: string]: number | string },
          objective_values: {} as { [key: string]: number }
        };

        // Update parameter names
        config.features.forEach(feature => {
          const oldValue = Object.entries(point.parameters).find(([key]) => 
            getStandardKey(feature.name) === getStandardKey(key)
          );
          if (oldValue) {
            newPoint.parameters[feature.name] = oldValue[1];
          }
        });

        // Update objective names
        config.objectives.forEach(objective => {
          const oldValue = Object.entries(point.objective_values).find(([key]) => 
            getStandardKey(objective.name) === getStandardKey(key)
          );
          if (oldValue) {
            newPoint.objective_values[objective.name] = oldValue[1];
          }
        });

        return newPoint;
      });

      setHistoricalData(updatedData);
    }
  }, [config]);

  // Get list of parameters
  const parameterNames = config.features.map(feature => feature.name);

  console.log('Parameter names:', parameterNames);

  const handleDataLoaded = async (data: { [key: string]: (number | string)[] }) => {
    try {
      // Use standardized keys for data processing
      const standardizedData = Object.entries(data).reduce((acc, [key, values]) => {
        const standardKey = getStandardKey(key);
        acc[standardKey] = values;
        return acc;
      }, {} as { [key: string]: (number | string)[] });

      console.log('Standardized data:', standardizedData);

      console.log('Original data:', Object.keys(data));
      console.log('Standardized data keys:', Object.keys(standardizedData));
      console.log('Standardized data values:', Object.values(standardizedData));

      const dataLength = Object.values(standardizedData)[0]?.length || 0;
      const transformedData = Array.from({ length: dataLength }, (_, index) => {
        const parameters: { [key: string]: number | string } = {};
        
        config.features.forEach(feature => {
          // Find the actual column name using lowercase matching
          const actualColumnName = getStandardKey(feature.name);
          
          switch (feature.type) {
            case 'continuous':
              if (actualColumnName && standardizedData[actualColumnName][index] !== undefined) {
                parameters[feature.name] = Number(standardizedData[actualColumnName][index]);
              }
              break;
            case 'discrete':
              if (actualColumnName && standardizedData[actualColumnName][index] !== undefined) {
                // Convert the value to lowercase to match API expectations
                const value = String(standardizedData[actualColumnName][index]).toLowerCase();
                // Only set if it matches one of the allowed categories
                if (feature.categories.includes(value)) {
                  parameters[feature.name] = value;
                } else {
                  console.warn(`Invalid value for ${feature.name}: ${value}. Expected one of: ${feature.categories.join(', ')}`);
                }
              }
              break;
            case 'composition':
              feature.columns.parts.forEach(columnName => {

                const actualPartName = getStandardKey(columnName);
                if (actualPartName && standardizedData[actualPartName][index] !== undefined) {
                  parameters[columnName] = Number(standardizedData[actualPartName][index]);
                }
              });
              break;
          }
        });
  
        const objective_values: { [key: string]: number } = {};
        config.objectives.forEach(objective => {
          const actualObjName = getStandardKey(objective.name);
          console.log('Objective mapping:', {
            original: objective.name,
            standardized: actualObjName,
            availableKeys: Object.keys(standardizedData),
            exists: standardizedData[actualObjName] !== undefined
          });
          
          if (actualObjName && standardizedData[actualObjName] !== undefined) {
            objective_values[objective.name] = Number(standardizedData[actualObjName][index]);
          } else {
            console.warn(`Missing data for objective: ${objective.name} (${actualObjName})`);
          }
        });
  
        return { parameters, objective_values };
      });

      console.log('Transformed data:', transformedData);
  
      // Validate transformed data
      if (transformedData.length === 0) {
        throw new Error("No valid data points found in file");
      }
  
      // Validate each data point has all required parameters
      const missingParams = transformedData.some(point => 
        config.features.some(feature => {
          switch (feature.type) {
            case 'continuous':
            case 'discrete':
              // Check direct feature name
              return point.parameters[feature.name] === undefined;
            case 'composition':
              // Check all part columns exist
              return feature.columns.parts.some(columnName => 
                point.parameters[columnName] === undefined
              );
            default:
              return true;
          }
        })
      );
      if (missingParams) {
        throw new Error("Some data points are missing required parameters or have invalid values");
      }

      // Validate each data point has all required objectives
      const missingObjectives = transformedData.some(point =>
        config.objectives.some(obj => {
          const value = point.objective_values[obj.name];
          return value === undefined || typeof value !== 'number';
        })
      );
  
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
      
      const processedData = historicalData?.map(point => ({
        parameters: point.parameters,
        objective_values: point.objective_values
      })) || [];

      const request: ActiveLearningRequest = {
        config,
        data: processedData,
        n_suggestions: numSuggestions
      };

      console.log('Full request details:', {
        configFeatures: config.features.map(f => ({
          name: f.name,
          type: f.type,
          ...(f.type === 'composition' ? { parts: f.columns.parts } : {})
        })),
        configObjectives: config.objectives.map(o => o.name),
        historicalDataSample: processedData[0],
        totalDataPoints: processedData.length,
        n_suggestions: numSuggestions
      });
      
      console.log('Raw request:', JSON.stringify(request, null, 2));
      
      const response = await getSuggestions(request);
      console.log('API Response:', response);
      
      setSuggestions(response.suggestions);
      setExpectedImprovements(null); // No longer using expected improvements
      toast.success("Generated new suggestions!");
    } catch (error) {
      console.error('API Error:', error);
      if (error instanceof Error && 'response' in error) {
        const axiosError = error as any;
        const errorDetail = axiosError.response?.data?.detail;
        console.error('Detailed API Error:', {
          status: axiosError.response?.status,
          statusText: axiosError.response?.statusText,
          data: axiosError.response?.data,
          validation: errorDetail,
          message: error.message
        });
        toast.error(`Error: ${errorDetail || error.message}`);
      } else {
        console.error('Unknown error type:', error);
        toast.error("Failed to generate suggestions");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const formatValue = (param: string, value: number | string) => {
    const feature = config.features.find(f => f.name === param);
    if (feature?.type === 'discrete') {
      return value; // string value for discrete parameters
    }
    return typeof value === 'number' ? value.toFixed(4) : value;
  };

  return (
    <div className="card p-6">
      <h2 className="text-2xl font-semibold mb-4">Active Learning</h2>
      <div className="space-y-6">
        <div className="flex items-center">
        </div>
        <div>
          <h3 className="text-lg font-medium mb-2">Upload Experimental Data</h3>
          <FileUpload onDataLoaded={handleDataLoaded} />
          {historicalData && (
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Loaded {historicalData.length} data points
            </p>
          )}
        </div>
        <div>
          <p className="text-lg font-medium mb-2">Upload Data based on the format provided above.</p>
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
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">{loadingMessage}</p>
          </div>
        )}

        {suggestions && (
          <div>
            <h3 className="text-lg font-medium mb-2">Suggested Experiments</h3>
            <div className="space-y-4">
              {suggestions.slice(0, numSuggestions).map((suggestion) => {
                console.log('Suggestion predictions:', suggestion.predictions);
                return (
                  <div key={suggestion.rank} className="bg-zinc-50 dark:bg-zinc-800 p-4 rounded-lg">
                    <div className="flex items-center gap-2 mb-3">
                      <h4 className="font-medium">Suggestion {suggestion.rank}</h4>
                      <span className="text-sm text-zinc-500 dark:text-zinc-400">
                        (Rank #{suggestion.rank})
                      </span>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-3">
                      {Object.entries(suggestion.suggestion).map(([param, value]) => {
                        if (value === undefined || value === null || value === '') {
                          return null;
                        }
                        
                        return (
                          <div key={param} className="bg-white dark:bg-zinc-900 p-3 rounded-md">
                            <p className="text-sm text-zinc-600 dark:text-zinc-400">{formatParamName(param)}</p>
                            <p className="font-mono">
                              {typeof value === 'number' 
                                ? value.toFixed(4)
                                : typeof value === 'object' && value !== null
                                  ? JSON.stringify(value)
                                  : String(value)
                              }
                            </p>
                          </div>
                        );
                      })}
                    </div>
                    <div className="bg-white/50 dark:bg-zinc-900/50 p-3 rounded-md mb-3">
                      <h5 className="text-sm font-medium mb-2 text-zinc-700 dark:text-zinc-300">Predicted Outcomes</h5>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                        {suggestion.predictions.map((predictionObj) => {
                          // Each prediction object contains one property and its std
                          const [[propName, value], [stdPropName, stdValue]] = Object.entries(predictionObj);
                          const baseName = propName.replace("_std", "");
                          
                          return (
                            <div key={baseName} className="bg-white/50 dark:bg-zinc-800/50 p-2 rounded">
                              <p className="text-xs text-zinc-600 dark:text-zinc-400">{formatParamName(baseName)}</p>
                              <p className="font-mono text-sm">
                                {typeof value === 'number' ? value.toFixed(4) : String(value)}
                                <span className="text-zinc-500"> ± {typeof stdValue === 'number' ? stdValue.toFixed(4) : String(stdValue)}</span>
                              </p>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                    <div className="bg-white/50 dark:bg-zinc-900/50 p-3 rounded-md">
                      <p className="text-sm text-zinc-700 dark:text-zinc-300">
                        {suggestion.reason}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 