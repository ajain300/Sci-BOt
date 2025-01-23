import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const API_BASE_URL = "http://localhost:8000";

export type ApiResponse<T> = {
  ok: boolean;
  data?: T;
  error?: string;
}; 