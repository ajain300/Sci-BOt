/**
 * Converts a snake_case string to Title Case
 * e.g., "monomer_composition" -> "Monomer Composition"
 */
export const formatParamName = (name: string): string => {
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

/**
 * You can add other string formatting utilities here as needed
 */ 