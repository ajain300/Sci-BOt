"use client";

import { useState, useEffect } from "react";
import { useToast } from "@/hooks/use-toast"
import { Toaster } from "@/components/ui/toaster"
import ActiveLearning from "@/components/ActiveLearning";
import CSVPreview from "@/components/CSVPreview";
import { getMockConfig } from "@/lib/mockData";
import type { OptimizationConfig } from "@/lib/api";
import { formatParamName } from "@/lib/utils/stringUtils";
import { api } from "@/lib/api";

export default function TestPage() {
  const [config, setConfig] = useState<OptimizationConfig | null>(null);
  const [isConfigAccepted, setIsConfigAccepted] = useState(false);
  const [editingField, setEditingField] = useState<{
    type: 'feature' | 'objective' | 'component';
    id: string;
    property: string;
    componentName?: string;  // For composition components
  } | null>(null);

  const { toast } = useToast()

  const handleValueUpdate = (
    id: string,
    property: string,
    value: number | string,
    type: 'feature' | 'objective' | 'component' = 'feature',
    componentName?: string
  ) => {
    try {
      setConfig(prevConfig => {
        if (!prevConfig) return prevConfig;
        
        const newConfig = JSON.parse(JSON.stringify(prevConfig));
        
        if (type === 'feature') {
          const feature = newConfig.features.find((f: { name: string }) => f.name === id);
          if (feature) {
            if (property === 'name') {
              // Validate name is unique
              const isNameTaken = newConfig.features.some((f: { name: string }) => f.name === value && f !== feature);
              if (isNameTaken) {
                throw new Error('Variable name must be unique');
              }
              feature.name = String(value);
            } else if (property === 'min' || property === 'max') {
              const numValue = Number(value);
              // Validate min/max relationships
              if (property === 'min' && numValue >= feature.max) {
                throw new Error('Minimum value must be less than maximum value');
              }
              if (property === 'max' && numValue <= feature.min) {
                throw new Error('Maximum value must be greater than minimum value');
              }
              feature[property] = numValue;
            } else if (property.includes('.')) {
              const numValue = Number(value);
              // Handle nested component ranges
              const [prop1, prop2, prop3, index] = property.split('.');
              if (prop1 === 'columns' && prop2 === 'range') {
                const rangeArray = feature.columns.range[prop3];
                // Validate min/max relationships for composition ranges
                if (Number(index) === 0 && numValue >= rangeArray[1]) {
                  throw new Error('Minimum value must be less than maximum value');
                }
                if (Number(index) === 1 && numValue <= rangeArray[0]) {
                  throw new Error('Maximum value must be greater than minimum value');
                }
                feature.columns.range[prop3][Number(index)] = numValue;
              }
            }
          }
        } else if (type === 'component' && componentName) {
          const feature = newConfig.features.find((f: { name: string }) => f.name === id);
          if (feature && feature.type === 'composition') {
            // Skip if name hasn't changed
            if (value === componentName) {
              return newConfig;
            }
            
            // Get the old range value
            const oldRange = feature.columns.range[componentName];
            
            // Validate new component name is unique
            if (feature.columns.range[value as string]) {
              throw new Error('Component name must be unique');
            }

            // Create new entry with new name and delete old one
            feature.columns.range[value as string] = oldRange;
            delete feature.columns.range[componentName];
          }
        } else if (type === 'objective') {
          const objective = newConfig.objectives.find((o: { name: string }) => o.name === id);
          if (objective && property === 'name') {
            // Validate name is unique among objectives
            const isNameTaken = newConfig.objectives.some((o: { name: string }) => o.name === value && o !== objective);
            if (isNameTaken) {
              throw new Error('Objective name must be unique');
            }
            objective.name = String(value);
          }
        }
        return newConfig;
      });

      toast({
        description: property === 'name' ? "Name updated successfully" : "Range updated successfully"
      });
    } catch (error) {
      console.error('Update error:', error);
      toast({
        variant: "destructive",
        description: error instanceof Error ? error.message : "Failed to update value"
      });
    }
  };

  // Update handlers to include component type
  const handleInputBlur = (
    e: React.FocusEvent<HTMLInputElement>, 
    id: string, 
    property: string, 
    type: 'feature' | 'objective' | 'component' = 'feature',
    componentName?: string
  ) => {
    const value = property === 'name' ? e.target.value : Number(e.target.value);
    if (property !== 'name' && isNaN(value as number)) {
      toast({
        variant: "destructive",
        description: "Please enter a valid number"
      });
      return;
    }
    handleValueUpdate(id, property, value, type, componentName);
    setEditingField(null);
  };

  const handleInputKeyDown = (
    e: React.KeyboardEvent<HTMLInputElement>, 
    id: string, 
    property: string, 
    type: 'feature' | 'objective' | 'component' = 'feature',
    componentName?: string
  ) => {
    if (e.key === 'Enter') {
      const value = property === 'name' ? e.currentTarget.value : Number(e.currentTarget.value);
      if (property !== 'name' && isNaN(value as number)) {
        toast({
          variant: "destructive",
          description: "Please enter a valid number"
        });
        return;
      }
      handleValueUpdate(id, property, value, type, componentName);
      setEditingField(null);
    } else if (e.key === 'Escape') {
      setEditingField(null);
    }
  };

  // Load mock config on component mount
  useEffect(() => {
    const mockConfig = getMockConfig();
    setConfig(mockConfig);
    setIsConfigAccepted(false);
    toast({
      description: "Mock configuration loaded for testing!"
    });
  }, []);

  const handleAcceptConfig = () => {
    setIsConfigAccepted(true);
    toast({
      description: "Configuration accepted! You can now proceed with the optimization."
    });
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <Toaster />
      <h1 className="text-4xl font-bold mb-8">Scientific Optimization (Test Mode)</h1>
      
      <div className="space-y-8">
        <div className="card p-6">
          <h2 className="text-2xl font-semibold mb-4">Mock Configuration</h2>
          <p className="mb-4">This page uses a mock configuration for testing without LLM prompts.</p>
        </div>

        {config && (
          <div className="card p-6">
            <h2 className="text-2xl font-semibold mb-4">Generated Configuration</h2>
            <p className="mb-4">This is how we've interpreted your problem in a more "machine-learnable" format, suitable for optimization. You can edit the variable names and ranges to customize the optimization search space.</p>
            <div className="space-y-4">
              <div>
                <h3 className="font-medium">Parameters</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {config.features.map((feature) => (
                    <div key={feature.name} className="bg-zinc-50 dark:bg-zinc-800 p-3 rounded-md">
                      <p className="font-medium">
                        <span 
                          className="cursor-pointer hover:bg-zinc-200 dark:hover:bg-zinc-700 px-1 rounded inline-block min-w-[100px]"
                          onDoubleClick={() => setEditingField({type: 'feature', id: feature.name, property: 'name'})}
                        >
                          {editingField?.type === 'feature' && 
                           editingField.id === feature.name && 
                           editingField.property === 'name' ? (
                            <input
                              type="text"
                              className="w-full bg-white dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded px-1"
                              defaultValue={feature.name}
                              autoFocus
                              onBlur={(e) => handleInputBlur(e, feature.name, 'name')}
                              onKeyDown={(e) => handleInputKeyDown(e, feature.name, 'name')}
                            />
                          ) : (
                            formatParamName(feature.name)
                          )}
                        </span>
                      </p>
                      <p className="text-zinc-600 dark:text-zinc-300">
                        Type: {feature.type}
                      </p>
                      {feature.type === 'continuous' && (
                        <p className="text-zinc-600 dark:text-zinc-400">
                          Range: [
                          <span 
                            className="cursor-pointer hover:bg-zinc-200 dark:hover:bg-zinc-700 px-1 rounded"
                            onDoubleClick={() => setEditingField({type: 'feature', id: feature.name, property: 'min'})}
                          >
                            {editingField?.type === 'feature' && 
                             editingField.id === feature.name && 
                             editingField.property === 'min' ? (
                              <input
                                type="number"
                                className="w-16 bg-white dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded"
                                defaultValue={feature.min}
                                autoFocus
                                onBlur={(e) => handleInputBlur(e, feature.name, 'min')}
                                onKeyDown={(e) => handleInputKeyDown(e, feature.name, 'min')}
                              />
                            ) : (
                              feature.min
                            )}
                          </span>, 
                          <span 
                            className="cursor-pointer hover:bg-zinc-200 dark:hover:bg-zinc-700 px-1 rounded"
                            onDoubleClick={() => setEditingField({type: 'feature', id: feature.name, property: 'max'})}
                          >
                            {editingField?.type === 'feature' && 
                             editingField.id === feature.name && 
                             editingField.property === 'max' ? (
                              <input
                                type="number"
                                className="w-16 bg-white dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded"
                                defaultValue={feature.max}
                                autoFocus
                                onBlur={(e) => handleInputBlur(e, feature.name, 'max')}
                                onKeyDown={(e) => handleInputKeyDown(e, feature.name, 'max')}
                              />
                            ) : (
                              feature.max
                            )}
                          </span>]
                        </p>
                      )}
                      {feature.type === 'discrete' && (
                        <p className="text-zinc-600 dark:text-zinc-300">
                          Options: {feature.categories.join(', ')}
                        </p>
                      )}
                      {feature.type === 'composition' && (
                        <>
                          <p className="text-zinc-600 dark:text-zinc-300 mb-2">
                            Components:
                          </p>
                          <div className="pl-4 space-y-2">
                            {Object.entries(feature.columns.range).map(([component, [min, max]]) => (
                              <div key={component} className="bg-zinc-100 dark:bg-zinc-700 p-2 rounded">
                                <p className="font-medium">
                                  <span 
                                    className="cursor-pointer hover:bg-zinc-200 dark:hover:bg-zinc-700 px-1 rounded inline-block min-w-[100px]"
                                    onDoubleClick={() => setEditingField({
                                      type: 'component', 
                                      id: feature.name, 
                                      property: 'name',
                                      componentName: component
                                    })}
                                  >
                                    {editingField?.type === 'component' && 
                                     editingField.id === feature.name && 
                                     editingField.property === 'name' &&
                                     editingField.componentName === component ? (
                                      <input
                                        type="text"
                                        className="w-full bg-white dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded px-1"
                                        defaultValue={component}
                                        autoFocus
                                        onBlur={(e) => handleInputBlur(e, feature.name, 'name', 'component', component)}
                                        onKeyDown={(e) => handleInputKeyDown(e, feature.name, 'name', 'component', component)}
                                      />
                                    ) : (
                                      formatParamName(component)
                                    )}
                                  </span>
                                </p>
                                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                                  Range: [
                                  <span 
                                    className="cursor-pointer hover:bg-zinc-200 dark:hover:bg-zinc-700 px-1 rounded"
                                    onDoubleClick={() => setEditingField({
                                      type: 'feature', 
                                      id: feature.name, 
                                      property: `columns.range.${component}.0`
                                    })}
                                  >
                                    {editingField?.type === 'feature' && 
                                     editingField.id === feature.name && 
                                     editingField.property === `columns.range.${component}.0` ? (
                                      <input
                                        type="number"
                                        className="w-14 bg-white dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded"
                                        defaultValue={min}
                                        autoFocus
                                        onBlur={(e) => handleInputBlur(e, feature.name, `columns.range.${component}.0`)}
                                        onKeyDown={(e) => handleInputKeyDown(e, feature.name, `columns.range.${component}.0`)}
                                      />
                                    ) : (
                                      min
                                    )}
                                  </span>, 
                                  <span 
                                    className="cursor-pointer hover:bg-zinc-200 dark:hover:bg-zinc-700 px-1 rounded"
                                    onDoubleClick={() => setEditingField({
                                      type: 'feature', 
                                      id: feature.name, 
                                      property: `columns.range.${component}.1`
                                    })}
                                  >
                                    {editingField?.type === 'feature' && 
                                     editingField.id === feature.name && 
                                     editingField.property === `columns.range.${component}.1` ? (
                                      <input
                                        type="number"
                                        className="w-14 bg-white dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded"
                                        defaultValue={max}
                                        autoFocus
                                        onBlur={(e) => handleInputBlur(e, feature.name, `columns.range.${component}.1`)}
                                        onKeyDown={(e) => handleInputKeyDown(e, feature.name, `columns.range.${component}.1`)}
                                      />
                                    ) : (
                                      max
                                    )}
                                  </span>]
                                </p>
                              </div>
                            ))}
                          </div>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="font-medium">Objectives</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
                  {config.objectives.map((objective, index) => (
                    <div key={index} className="bg-zinc-50 dark:bg-zinc-800 p-3 rounded-md">
                      <p className="font-medium text-zinc-800 dark:text-zinc-200">
                        <span 
                          className="cursor-pointer hover:bg-zinc-200 dark:hover:bg-zinc-700 px-1 rounded inline-block min-w-[100px]"
                          onDoubleClick={() => setEditingField({type: 'objective', id: objective.name, property: 'name'})}
                        >
                          {editingField?.type === 'objective' && 
                           editingField.id === objective.name && 
                           editingField.property === 'name' ? (
                            <input
                              type="text"
                              className="w-full bg-white dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-600 rounded px-1"
                              defaultValue={objective.name}
                              autoFocus
                              onBlur={(e) => handleInputBlur(e, objective.name, 'name', 'objective')}
                              onKeyDown={(e) => handleInputKeyDown(e, objective.name, 'name', 'objective')}
                            />
                          ) : (
                            formatParamName(objective.name)
                          )}
                        </span>
                      </p>
                      <div className="mt-2 space-y-1">
                        <p className="text-sm text-zinc-600 dark:text-zinc-400">
                          <span className="font-medium">Direction:</span> {objective.direction}
                        </p>
                        {objective.weight && (
                          <p className="text-sm text-zinc-600 dark:text-zinc-400">
                            <span className="font-medium">Weight:</span> {objective.weight}
                          </p>
                        )}
                        {objective.target_value && (
                          <p className="text-sm text-zinc-600 dark:text-zinc-400">
                            <span className="font-medium">Target:</span> {objective.target_value}
                          </p>
                        )}
                        {objective.range_min !== undefined && 
                         objective.range_max !== undefined && 
                         objective.range_min !== null && 
                         objective.range_max !== null && (
                          <p className="text-sm text-zinc-600 dark:text-zinc-400">
                            <span className="font-medium">Range:</span> [{objective.range_min}, {objective.range_max}]
                          </p>
                        )}
                      </div>
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

              <div className="flex justify-end mt-6">
                <button 
                  onClick={handleAcceptConfig}
                  disabled={isConfigAccepted}
                  className="px-4 py-2 bg-zinc-900 text-zinc-50 rounded-md hover:bg-zinc-800 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isConfigAccepted ? "Configuration Accepted" : "Accept Configuration"}
                </button>
              </div>
            </div>
          </div>
        )}

        {config && isConfigAccepted && (
          <>
            <CSVPreview config={config} />
            <ActiveLearning config={config} />
          </>
        )}
      </div>
    </div>
  );
}
