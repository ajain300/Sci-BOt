"use client";

import { useState } from "react";
import { toast } from "react-hot-toast";
import { analyzeData } from "@/lib/api";
import type { OptimizationConfig, ProcessDataResponse } from "@/lib/api";
import FileUpload from "./FileUpload";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface Props {
  config: OptimizationConfig;
}

export default function DataAnalysis({ config }: Props) {
  const [analysis, setAnalysis] = useState<ProcessDataResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleDataLoaded = async (data: { [key: string]: number[] }) => {
    try {
      setIsLoading(true);
      const response = await analyzeData({
        config,
        data
      });
      setAnalysis(response);
      toast.success("Analysis complete!");
    } catch (error) {
      toast.error("Failed to analyze data");
    } finally {
      setIsLoading(false);
    }
  };

  const prepareChartData = (data: { [key: string]: number[] }) => {
    const maxLength = Math.max(...Object.values(data).map(arr => arr.length));
    return Array.from({ length: maxLength }, (_, i) => ({
      index: i + 1,
      ...Object.fromEntries(
        Object.entries(data).map(([key, values]) => [key, values[i]])
      ),
    }));
  };

  return (
    <div className="card p-6">
      <h2 className="text-2xl font-semibold mb-4">Data Analysis</h2>
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-medium mb-2">Upload Experiment Data</h3>
          <FileUpload onDataLoaded={handleDataLoaded} />
        </div>

        {isLoading && (
          <div className="text-center py-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">Analyzing data...</p>
          </div>
        )}

        {analysis && (
          <>
            <div>
              <h3 className="text-lg font-medium mb-2">Best Result</h3>
              <div className="bg-zinc-50 dark:bg-zinc-800 p-4 rounded-lg">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {Object.entries(analysis.best_point.parameters).map(([param, value]) => (
                    <div key={param} className="bg-white dark:bg-zinc-900 p-3 rounded-md">
                      <p className="text-sm text-zinc-600 dark:text-zinc-400">{param}</p>
                      <p className="font-mono">{value.toFixed(4)}</p>
                    </div>
                  ))}
                  <div className="bg-white dark:bg-zinc-900 p-3 rounded-md">
                    <p className="text-sm text-zinc-600 dark:text-zinc-400">{config.objective_variable}</p>
                    <p className="font-mono">{analysis.best_point.objective_value?.toFixed(4)}</p>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium mb-2">Parameter Importance</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {Object.entries(analysis.parameter_importance).map(([param, importance]) => (
                  <div key={param} className="bg-zinc-50 dark:bg-zinc-800 p-3 rounded-md">
                    <p className="text-sm text-zinc-600 dark:text-zinc-400">{param}</p>
                    <div className="mt-2 h-4 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500"
                        style={{ width: `${importance * 100}%` }}
                      />
                    </div>
                    <p className="text-xs text-zinc-500 mt-1">{(importance * 100).toFixed(1)}%</p>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium mb-2">Optimization Progress</h3>
              <div className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={prepareChartData(analysis.data)}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#666" />
                    <XAxis 
                      dataKey="index" 
                      label={{ value: "Iteration", position: "bottom" }}
                      stroke="#666"
                    />
                    <YAxis stroke="#666" />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgb(var(--background-rgb))',
                        border: '1px solid #666'
                      }}
                    />
                    <Legend />
                    {Object.keys(analysis.data).map((param, index) => (
                      <Line
                        key={param}
                        type="monotone"
                        dataKey={param}
                        stroke={`hsl(${index * 360 / Object.keys(analysis.data).length}, 70%, 50%)`}
                        dot={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
} 