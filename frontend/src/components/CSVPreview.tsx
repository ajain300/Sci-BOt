"use client";

import { useState, useEffect } from "react";
import { getCSVTemplate } from "@/lib/api";
import type { OptimizationConfig } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

interface Props {
  config: OptimizationConfig;
}

export default function CSVPreview({ config }: Props) {
  const [headers, setHeaders] = useState<string[]>([]);
  const { toast } = useToast();

  useEffect(() => {
    const fetchTemplate = async () => {
      try {
        const content = await getCSVTemplate(config);
        // Parse the CSV content to get headers
        const headers = content.trim().split(',');
        setHeaders(headers);
      } catch (error) {
        toast({
          title: "Error",
          description: "Failed to generate CSV template",
          variant: "destructive"
        });
      }
    };
    fetchTemplate();
  }, [config, toast]);

  const handleDownload = () => {
    // Create CSV content from headers
    const csvContent = headers.join(',') + '\n';
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "optimization_template.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="card p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-semibold">Data Template</h2>
        <button onClick={handleDownload} className="btn-primary">
          Download CSV
        </button>
      </div>
      
      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          <div className="border border-zinc-200 dark:border-zinc-700 rounded-lg">
            <table className="min-w-full divide-y divide-zinc-200 dark:divide-zinc-700">
              <thead>
                <tr>
                  {headers.map((header, i) => (
                    <th
                      key={i}
                      className="px-4 py-3 bg-zinc-50 dark:bg-zinc-800 text-left text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider border-r border-zinc-200 dark:border-zinc-700 last:border-r-0"
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-zinc-900 divide-y divide-zinc-200 dark:divide-zinc-700">
                {/* Empty rows for example */}
                {[...Array(3)].map((_, rowIndex) => (
                  <tr key={rowIndex}>
                    {headers.map((_, colIndex) => (
                      <td
                        key={colIndex}
                        className="px-4 py-3 whitespace-nowrap text-sm text-zinc-400 dark:text-zinc-500 border-r border-zinc-200 dark:border-zinc-700 last:border-r-0"
                      >
                        ...
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
} 