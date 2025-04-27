const { 
  createSeededRandom
} = require('../utils/dataLoader');

const MASTER_SEED = 10;
const seededRandom = createSeededRandom(MASTER_SEED);

class LinearRegression {
    constructor() {
      this.weights = null;
      this.bias = null;
    }
    
    train(features, target, learningRate = 0.01, iterations = 1000) {
      const n = features.length;
      const numFeatures = features[0].length;
      
      // 初始化权重和偏置
      this.weights = Array(numFeatures).fill(0).map(() => seededRandom() * 0.1);
      this.bias = 0;
      
      // 梯度下降优化
      for (let iter = 0; iter < iterations; iter++) {
        // 计算预测值
        const predictions = features.map(x => this.predict(x));
        
        // 计算梯度
        const weightGradients = Array(numFeatures).fill(0);
        let biasGradient = 0;
        
        for (let i = 0; i < n; i++) {
          const error = predictions[i] - target[i];
          
          for (let j = 0; j < numFeatures; j++) {
            weightGradients[j] += (error * features[i][j]) / n;
          }
          
          biasGradient += error / n;
        }
        
        // 更新权重和偏置
        for (let j = 0; j < numFeatures; j++) {
          this.weights[j] -= learningRate * weightGradients[j];
        }
        this.bias -= learningRate * biasGradient;
      }
      
      return this;
    }
    
    predict(x) {
      let prediction = this.bias;
      
      for (let i = 0; i < this.weights.length; i++) {
        prediction += this.weights[i] * x[i];
      }
      
      return prediction;
    }
    
    score(features, target) {
      const predictions = features.map(x => this.predict(x));
      
      // 计算R²
      const meanTarget = target.reduce((sum, val) => sum + val, 0) / target.length;
      const totalSumSquares = target.reduce((sum, val) => sum + Math.pow(val - meanTarget, 2), 0);
      const residualSumSquares = target.reduce((sum, val, i) => sum + Math.pow(val - predictions[i], 2), 0);
      
      return 1 - (residualSumSquares / totalSumSquares);
    }
    
    getFeatureImportance() {
      const absWeights = this.weights.map(w => Math.abs(w));
      const totalWeight = absWeights.reduce((sum, w) => sum + w, 0);
      
      return absWeights.map(w => (w / totalWeight) * 100);
    }
  }
  
  module.exports = LinearRegression;