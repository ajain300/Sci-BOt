"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { toast } from "react-hot-toast";
import { getSuggestions } from "@/lib/api";
import type { OptimizationConfig, ActiveLearningResponse } from "@/lib/api";

const formSchema = z.object({
  n_suggestions: z.number().min(1).max(10),
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

export default function ActiveLearning({ config }: Props) {
  const [suggestions, setSuggestions] = useState<ActiveLearningResponse | null>(null);
  const { register, handleSubmit, formState: { errors } } = useForm<FormData>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      n_suggestions: 3,
      data: "{}",
    },
  });

  const onSubmit = async (data: FormData) => {
    try {
      const response = await getSuggestions({
        config,
        data: data.data,
        n_suggestions: data.n_suggestions,
      });
      setSuggestions(response);
      toast.success("Generated new suggestions!");
    } catch (error) {
      toast.error("Failed to generate suggestions");
    }
  };

  return (
    <div className="space-y-6">
      <div className="card p-6">
        <h2 className="text-2xl font-semibold mb-4">Active Learning</h2>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Number of Suggestions
            </label>
            <input
              type="number"
              {...register("n_suggestions", { valueAsNumber: true })}
              className="input-base w-full p-2 rounded-md"
              min={1}
              max={10}
            />
            {errors.n_suggestions && (
              <p className="text-red-500 text-sm mt-1">{errors.n_suggestions.message as string}</p>
            )}
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">
              Previous Data (JSON format)
            </label>
            <textarea
              {...register("data")}
              className="input-base w-full h-32 p-2 rounded-md font-mono"
              placeholder='{
  "parameter1": [1.2, 2.3, 3.4],
  "parameter2": [0.1, 0.2, 0.3]
}'
            />
            {typeof errors.data?.message === 'string' && (
              <p className="text-red-500 text-sm mt-1">{errors.data.message}</p>
            )}
          </div>

          <button type="submit" className="btn-primary">
            Generate Suggestions
          </button>
        </form>
      </div>

      {suggestions && (
        <div className="card p-6">
          <h2 className="text-2xl font-semibold mb-4">Suggestions</h2>
          <div className="space-y-4">
            {Object.entries(suggestions.suggestions).map(([param, values], paramIndex) => (
              <div key={param}>
                <h3 className="font-medium">{param}</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mt-2">
                  {values.map((value, valueIndex) => (
                    <div
                      key={valueIndex}
                      className="bg-zinc-50 dark:bg-zinc-800 p-3 rounded-md"
                    >
                      <p className="text-sm text-zinc-600 dark:text-zinc-300">
                        Suggestion {valueIndex + 1}
                      </p>
                      <p className="font-mono">
                        {value.toFixed(4)}
                      </p>
                      <p className="text-xs text-zinc-500 dark:text-zinc-400">
                        Uncertainty: {suggestions.uncertainty[paramIndex].toFixed(4)}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 