// test/dataProcessor.test.js
const { 
    loadData, 
    calculateCompositeScores, 
    prepareMLDataset, 
    extractFeaturesAndTarget, 
    trainTestSplit, 
    constructDefinitions 
  } = require('../src/utils/dataLoader.js');
  
  const fs = require('fs');
  const path = require('path');
  
  // Mock fs and path modules
  jest.mock('fs');
  jest.mock('path');
  
  describe('Data Processing Module Tests', () => {
    
    // Test loadData function
    describe('loadData', () => {
      beforeEach(() => {
        // Reset mocks before each test
        jest.clearAllMocks();
      });
  
      it('should parse CSV data correctly', () => {
        // Stub for fs.readFileSync
        const mockCsvContent = 'CPT1,CPT2,RAD1\n4,5,3\n5,3,4';
        fs.readFileSync.mockReturnValue(mockCsvContent);
        path.resolve.mockReturnValue('/mocked/path');
  
        const result = loadData('dummy.csv');
        
        expect(fs.readFileSync).toHaveBeenCalledWith('/mocked/path', 'utf8');
        expect(result).toEqual([
          { CPT1: 4, CPT2: 5, RAD1: 3 },
          { CPT1: 5, CPT2: 3, RAD1: 4 }
        ]);
      });
  
      it('should handle empty CSV files', () => {
        fs.readFileSync.mockReturnValue('CPT1,CPT2\n');
        const result = loadData('empty.csv');
        expect(result).toEqual([]);
      });
  
      it('should handle CSV parsing errors gracefully', () => {
        // Create a CSV with inconsistent columns
        fs.readFileSync.mockReturnValue('CPT1,CPT2\n1,2,3\n4,5');
        
        // Mock console.error to prevent test output pollution
        console.error = jest.fn();
        
        const result = loadData('bad.csv');
        expect(console.error).toHaveBeenCalled();
      });
    });
  
    // Test calculateCompositeScores function
    describe('calculateCompositeScores', () => {
      it('should calculate scores correctly', () => {
        const mockData = [
          { CPT1: 5, CPT2: 4, CPT3: 3, RAD1: 5, RAD2: 4 },
          { CPT1: 3, CPT2: 3, CPT3: 3, RAD1: 4, RAD2: 3 }
        ];
  
        const result = calculateCompositeScores(mockData);
        
        expect(result.CPT[0]).toBeCloseTo(4); // (5+4+3)/3 = 4
        expect(result.CPT[1]).toBeCloseTo(3); // (3+3+3)/3 = 3
        expect(result.RAD[0]).toBeCloseTo(4.5); // (5+4)/2 = 4.5
        expect(result.RAD[1]).toBeCloseTo(3.5); // (4+3)/2 = 3.5
      });
  
      it('should handle missing values', () => {
        const mockData = [
          { CPT1: 5, CPT2: null, CPT3: 3 },
          { CPT1: 3, CPT3: 3 }  // CPT2 is undefined
        ];
  
        const result = calculateCompositeScores(mockData);
        
        expect(result.CPT[0]).toBeCloseTo(4); // (5+3)/2 = 4, ignoring null
        expect(result.CPT[1]).toBeCloseTo(3); // (3+3)/2 = 3, ignoring undefined
      });
  
      it('should return null when all values for a construct are missing', () => {
        const mockData = [
          { CPT1: null, CPT2: null, CPT3: null },
          { RAD1: 5, RAD2: 4 }
        ];
  
        const result = calculateCompositeScores(mockData);
        
        expect(result.CPT[0]).toBeNull();
        expect(result.RAD[0]).toBeCloseTo(4.5);
      });
    });
  
    // Test prepareMLDataset function
    describe('prepareMLDataset', () => {
      it('should create a valid ML dataset', () => {
        const mockCompositeScores = {
          CPT: [4, 3, null],
          RAD: [4.5, 3.5, 5],
          BI: [5, 4, null]
        };
  
        const result = prepareMLDataset(mockCompositeScores);
        
        // Only rows with non-null values should be included
        expect(result.length).toBe(2);
        expect(result[0]).toEqual({ CPT: 4, RAD: 4.5, BI: 5 });
        expect(result[1]).toEqual({ CPT: 3, RAD: 3.5, BI: 4 });
      });
  
      it('should handle empty input', () => {
        const mockCompositeScores = {
          CPT: [],
          RAD: [],
          BI: []
        };
  
        const result = prepareMLDataset(mockCompositeScores);
        expect(result).toEqual([]);
      });
    });
  
    // Test extractFeaturesAndTarget function
    describe('extractFeaturesAndTarget', () => {
      it('should extract features and target correctly', () => {
        const mockDataset = [
          { CPT: 4, RAD: 4.5, BI: 5 },
          { CPT: 3, RAD: 3.5, BI: 4 }
        ];
  
        const { features, target } = extractFeaturesAndTarget(
          mockDataset, 
          ['CPT', 'RAD'], 
          'BI'
        );
        
        expect(features).toEqual([[4, 4.5], [3, 3.5]]);
        expect(target).toEqual([5, 4]);
      });
    });
  
    // Test trainTestSplit function
    describe('trainTestSplit', () => {
      it('should split data into train and test sets with correct proportions', () => {
        const mockFeatures = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]];
        const mockTarget = [1, 2, 3, 4, 5];
        
        // For predictable testing, let's mock Math.random
        const originalRandom = Math.random;
        Math.random = jest.fn().mockReturnValue(0.5); // Always returns 0.5
        
        const result = trainTestSplit(mockFeatures, mockTarget, 0.4);
        
        // Restore original Math.random
        Math.random = originalRandom;
        
        // Test set should be 40% of the data (2 items)
        expect(result.testFeatures.length).toBe(2);
        expect(result.testTarget.length).toBe(2);
        
        // Train set should be 60% of the data (3 items)
        expect(result.trainFeatures.length).toBe(3);
        expect(result.trainTarget.length).toBe(3);
      });
      
      it('should handle edge cases with very small datasets', () => {
        const mockFeatures = [[1, 2]];
        const mockTarget = [1];
        
        const result = trainTestSplit(mockFeatures, mockTarget, 0.2);
        
        // With just one sample, it should still create train/test splits
        expect(result.trainFeatures.length + result.testFeatures.length).toBe(1);
      });
    });
  });