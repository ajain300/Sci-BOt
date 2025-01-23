"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { toast } from "react-hot-toast";
import { analyzeData } from "@/lib/api";
import type { OptimizationConfig, ProcessDataResponse } from "@/lib/api";
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

const formSchema = z.object({
  data: z.record(z.string(), z.array(z.number())).transform((val, ctx) => {
    try {
      return typeof val === "string" ? JSON.parse(val) : val;
    } catch {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Invalid JSON format",
      });
      return z.NEVER;
    }
  }),
});

type FormData = z.infer<typeof formSchema>;

interface Props {
  config: OptimizationConfig;
}

export default function DataAnalysis({ config }: Props) {
  const [analysis, setAnalysis] = useState<ProcessDataResponse | null>(null);
  const { register, handleSubmit, formState: { errors } } = useForm<FormData>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      data: "{}",
    },
  });

  const onSubmit = async (data: FormData) => {
    try {
      const response = await analyzeData({
        config,
        data: data.data,
      });
      setAnalysis(response);
      toast.success("Analysis completed!");
    } catch (error) {
      toast.error("Failed to analyze data");
    }
  };

  const prepareChartData = (data: Record<string, number[]>) => {
    const chartData = [];
    const numPoints = Object.values(data)[0]?.length || 0;
    
    for (let i = 0; i < numPoints; i++) {
      const point: Record<string, any> = { index: i };
      Object.entries(data).forEach(([param, values]) => {
        point[param] = values[i];
      });
      chartData.push(point);
    }
    
    return chartData;
  };

  const getMonochromaticColor = (index: number) => {
    const baseHue = 0; // 0 for red, can be adjusted
    const saturation = 0; // 0% for grayscale
    const lightness = Math.max(20, 90 - (index * 15)); // Decreasing lightness
    return `hsl(${baseHue}, ${saturation}%, ${lightness}%)`;
  };

  return (
    <div className="space-y-6">
      <div className="card p-6">
        <h2 className="text-2xl font-semibold mb-4">Data Analysis</h2>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Optimization Data (JSON format)
            </label>
            <textarea
              {...register("data")}
              className="input-base w-full h-32 p-2 rounded-md font-mono"
              placeholder='{
  "parameter1": [1.2, 2.3, 3.4],
  "parameter2": [0.1, 0.2, 0.3],
  "objective": [-1.5, -2.1, -1.8]
}'
            />
            {typeof errors.data?.message === 'string' && (
              <p className="text-red-500 text-sm mt-1">{errors.data.message}</p>
            )}
          </div>

          <button type="submit" className="btn-primary">
            Analyze Data
          </button>
        </form>
      </div>

      {analysis && (
        <div className="space-y-6">
          <div className="card p-6">
            <h2 className="text-2xl font-semibold mb-4">Analysis Results</h2>
            <div className="space-y-4">
              <div>
                <h3 className="font-medium">Best Point Found</h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mt-2">
                  {Object.entries(analysis.best_point).map(([param, value]) => (
                    <div key={param} className="bg-zinc-50 dark:bg-zinc-800 p-3 rounded-md">
                      <p className="text-sm font-medium">{param}</p>
                      <p className="font-mono">{value.toFixed(4)}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="font-medium">Performance Metrics</h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mt-2">
                  {Object.entries(analysis.performance_metrics).map(([metric, value]) => (
                    <div key={metric} className="bg-zinc-50 dark:bg-zinc-800 p-3 rounded-md">
                      <p className="text-sm font-medium">{metric}</p>
                      <p className="font-mono">{value.toFixed(4)}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="font-medium">Insights</h3>
                <ul className="list-disc list-inside space-y-2 mt-2">
                  {analysis.insights.map((insight, index) => (
                    <li key={index} className="text-zinc-600 dark:text-zinc-300">{insight}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          <div className="card p-6">
            <h2 className="text-2xl font-semibold mb-4">Optimization Progress</h2>
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
                      stroke={getMonochromaticColor(index)}
                      dot={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 