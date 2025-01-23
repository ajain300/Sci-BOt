"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { toast } from "react-hot-toast";
import { generateConfig } from "@/lib/api";
import type { OptimizationConfig } from "@/lib/api";
import ActiveLearning from "@/components/ActiveLearning";
import DataAnalysis from "@/components/DataAnalysis";

const formSchema = z.object({
  prompt: z.string().min(10, "Prompt must be at least 10 characters"),
});

type FormData = z.infer<typeof formSchema>;

export default function Home() {
  const [config, setConfig] = useState<OptimizationConfig | null>(null);
  const { register, handleSubmit, formState: { errors } } = useForm<FormData>({
    resolver: zodResolver(formSchema),
  });

  const onSubmit = async (data: FormData) => {
    try {
      const config = await generateConfig(data.prompt);
      setConfig(config);
      toast.success("Configuration generated successfully!");
    } catch (error) {
      toast.error("Failed to generate configuration");
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8">Scientific Optimization</h1>
      
      <div className="space-y-8">
        <div className="card p-6">
          <h2 className="text-2xl font-semibold mb-4">Generate Configuration</h2>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Describe your optimization problem
              </label>
              <textarea
                {...register("prompt")}
                className="input-base w-full h-32 p-2 rounded-md"
                placeholder="e.g., Optimize the efficiency of a solar panel by adjusting tilt angle and material thickness..."
              />
              {errors.prompt && (
                <p className="text-red-500 text-sm mt-1">{errors.prompt.message as string}</p>
              )}
            </div>
            <button type="submit" className="btn-primary">
              Generate Configuration
            </button>
          </form>
        </div>

        {config && (
          <div className="card p-6">
            <h2 className="text-2xl font-semibold mb-4">Generated Configuration</h2>
            <div className="space-y-4">
              <div>
                <h3 className="font-medium">Objective Function</h3>
                <p className="text-zinc-600 dark:text-zinc-300">{config.objective}</p>
              </div>
              <div>
                <h3 className="font-medium">Parameters</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {Object.entries(config.parameters).map(([name, param]) => (
                    <div key={name} className="bg-zinc-50 dark:bg-zinc-800 p-3 rounded-md">
                      <p className="font-medium">{name}</p>
                      <p className="text-zinc-600 dark:text-zinc-300">
                        Range: [{param.min}, {param.max}]
                      </p>
                      <p className="text-zinc-600 dark:text-zinc-300">
                        Type: {param.type}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
              {config.constraints && config.constraints.length > 0 && (
                <div>
                  <h3 className="font-medium">Constraints</h3>
                  <ul className="list-disc list-inside text-zinc-600 dark:text-zinc-300">
                    {config.constraints.map((constraint, i) => (
                      <li key={i}>{constraint}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}

        {config && (
          <>
            <ActiveLearning config={config} />
            <DataAnalysis config={config} />
          </>
        )}
      </div>
    </div>
  );
}
