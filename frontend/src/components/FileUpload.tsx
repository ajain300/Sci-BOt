"use client";

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import * as XLSX from 'xlsx';
import { toast } from 'react-hot-toast';

interface Props {
  onDataLoaded: (data: { [key: string]: number[] }) => void;
  accept?: string;
}

interface ExcelRow {
  [key: string]: string | number;
}

export default function FileUpload({ onDataLoaded, accept = ".csv,.xlsx,.tsv" }: Props) {
  const [fileName, setFileName] = useState<string>("");

  const processFile = async (file: File) => {
    try {
      console.log('Processing file:', file.name);
      const data: { [key: string]: number[] } = {};
      
      if (file.name.endsWith('.csv') || file.name.endsWith('.tsv')) {
        const text = await file.text();
        console.log('Raw file content:', text);
        
        // Try to detect if it's tab-separated by checking for tabs
        const separator = text.includes('\t') ? '\t' : ',';
        console.log('Detected separator:', separator === '\t' ? 'TAB' : 'COMMA');
        
        const rows = text.trim().split('\n').map(row => row.split(separator));
        console.log('Split rows:', rows);
        
        // Clean up headers (remove any \r and trim whitespace)
        const headers = rows[0].map(header => header.trim().replace(/\r$/, ''));
        console.log('Cleaned headers:', headers);
        
        // Initialize arrays for each column
        headers.forEach(header => {
          data[header] = [];
        });
        
        // Parse data rows
        for (let i = 1; i < rows.length; i++) {
          const rowData = rows[i].map(cell => cell.trim());
          console.log(`Processing row ${i}:`, rowData);
          rowData.forEach((value, j) => {
            const parsedValue = parseFloat(value);
            console.log(`Parsing value for ${headers[j]}:`, value, '→', parsedValue);
            if (!isNaN(parsedValue)) {
              data[headers[j]].push(parsedValue);
            }
          });
        }
      } else if (file.name.endsWith('.xlsx')) {
        const buffer = await file.arrayBuffer();
        const workbook = XLSX.read(buffer);
        const worksheet = workbook.Sheets[workbook.SheetNames[0]];
        const jsonData = XLSX.utils.sheet_to_json<ExcelRow>(worksheet);
        console.log('Parsed Excel data:', jsonData);
        
        if (jsonData.length === 0) {
          throw new Error("No data found in Excel file");
        }
        
        // Get headers from first row and clean them
        const headers = Object.keys(jsonData[0]).map(header => header.trim());
        console.log('Excel headers:', headers);
        
        // Initialize arrays for each column
        headers.forEach(header => {
          data[header] = [];
        });
        
        // Parse data rows
        jsonData.forEach((row: ExcelRow) => {
          console.log('Processing Excel row:', row);
          headers.forEach(header => {
            const cellValue = row[header];
            const value = typeof cellValue === 'number' 
              ? cellValue 
              : parseFloat(String(cellValue).trim());
            console.log(`Parsing Excel value for ${header}:`, cellValue, '→', value);
            if (!isNaN(value)) {
              data[header].push(value);
            }
          });
        });
      }

      // Validate that we have data
      if (Object.keys(data).length === 0) {
        throw new Error("No valid data found in file");
      }

      // Validate all arrays have the same length
      const lengths = Object.values(data).map(arr => arr.length);
      if (!lengths.every(len => len === lengths[0])) {
        throw new Error("Inconsistent data: some columns have missing values");
      }

      console.log('Final processed data:', data);
      onDataLoaded(data);
      setFileName(file.name);
      toast.success("File loaded successfully!");
    } catch (error) {
      console.error('Error processing file:', error);
      toast.error(error instanceof Error ? error.message : "Error processing file. Please check the format.");
    }
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      processFile(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'text/tab-separated-values': ['.tsv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    },
    maxFiles: 1
  });

  return (
    <div 
      {...getRootProps()} 
      className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
        ${isDragActive 
          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
          : 'border-zinc-300 dark:border-zinc-700 hover:border-blue-500 dark:hover:border-blue-500'
        }`}
    >
      <input {...getInputProps()} />
      <div className="space-y-2">
        <div className="text-3xl">📄</div>
        {fileName ? (
          <>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">Current file: {fileName}</p>
            <p className="text-xs text-zinc-500 dark:text-zinc-500">Drop a new file to replace</p>
          </>
        ) : (
          <>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">
              {isDragActive ? 'Drop the file here' : 'Drag & drop a file here, or click to select'}
            </p>
            <p className="text-xs text-zinc-500 dark:text-zinc-500">Supports CSV, TSV, and XLSX files</p>
          </>
        )}
      </div>
    </div>
  );
} 