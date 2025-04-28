const { 
  createSeededRandom
} = require('../utils/dataLoader');

const MASTER_SEED = 10;
const seededRandom = createSeededRandom(MASTER_SEED);

class RandomForestRegressor {
    constructor(numTrees = 10, maxDepth = 5, minSamplesPerTree = 0.8, featureSubsetRatio = 0.7) {
      this.numTrees = numTrees;
      this.maxDepth = maxDepth;
      this.trees = [];
      this.minSamplesPerTree = minSamplesPerTree; // Ratio of samples to use for each tree
      this.featureSubsetRatio = featureSubsetRatio; // Ratio of features to consider for each split
    }
    
    // Bootstrap sampling - randomly sample with replacement
    bootstrapSample(features, target) {
      const n = features.length;
      const sampleSize = Math.round(n * this.minSamplesPerTree);
      const sampleIndices = [];
      
      for (let i = 0; i < sampleSize; i++) {
        const randomIndex = Math.floor(seededRandom() * n);
        sampleIndices.push(randomIndex);
      }
      
      const sampleFeatures = sampleIndices.map(idx => features[idx]);
      const sampleTarget = sampleIndices.map(idx => target[idx]);
      
      return { sampleFeatures, sampleTarget };
    }
    
    // Get random subset of feature indices
    getFeatureSubset(numFeatures) {
      const subsetSize = Math.max(1, Math.round(numFeatures * this.featureSubsetRatio));
      const indices = Array.from({ length: numFeatures }, (_, i) => i);
      
      // Shuffle array
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(seededRandom() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      
      return indices.slice(0, subsetSize);
    }
    
    train(features, target) {
      const DecisionTreeRegressor = require('./decisionTree');
      
      const numFeatures = features[0].length;
      
      for (let i = 0; i < this.numTrees; i++) {
        // Create bootstrap sample
        const { sampleFeatures, sampleTarget } = this.bootstrapSample(features, target);
        
        // Create tree with feature subset selection capability
        const tree = new DecisionTreeRegressor(this.maxDepth);
        
        // Store feature subset in tree for prediction time
        tree.featureSubset = this.getFeatureSubset(numFeatures);
        
        // Use this subset for training by creating a filtered dataset
        const filteredFeatures = sampleFeatures.map(sample => 
          tree.featureSubset.map(featureIdx => sample[featureIdx])
        );
        
        // Train the tree on bootstrap sample
        tree.train(filteredFeatures, sampleTarget);
        
        // Add tree to forest
        this.trees.push(tree);
      }
      
      return this;
    }
    
    predictSingle(sample) {
      if (this.trees.length === 0) {
        throw new Error("Model not trained yet");
      }
      
      // Get predictions from all trees
      const predictions = this.trees.map(tree => {
        // Filter the sample to only include features used by this tree
        const filteredSample = tree.featureSubset.map(featureIdx => sample[featureIdx]);
        return tree.predictSingle(filteredSample);
      });
      
      // Average the predictions
      const sum = predictions.reduce((acc, val) => acc + val, 0);
      return sum / predictions.length;
    }
    
    predict(features) {
      return features.map(sample => this.predictSingle(sample));
    }
    
    score(features, target) {
      const predictions = this.predict(features);
      
      // Calculate mean of target values
      const meanTarget = target.reduce((sum, val) => sum + val, 0) / target.length;
      
      // Calculate total sum of squares
      const totalSumSquares = target.reduce((sum, val) => sum + Math.pow(val - meanTarget, 2), 0);
      
      // Calculate residual sum of squares
      const residualSumSquares = target.reduce((sum, val, i) => sum + Math.pow(val - predictions[i], 2), 0);
      
      // R-squared = 1 - (residual sum of squares / total sum of squares)
      return 1 - (residualSumSquares / totalSumSquares);
    }
    
    getFeatureImportance(numFeatures) {
      if (this.trees.length === 0) {
        throw new Error("Model not trained yet");
      }
      
      // 初始化重要性数组
      const importance = Array(numFeatures).fill(0);
      const featureCounts = Array(numFeatures).fill(0);
      
      // 使用基于树结构的重要性计算
      this.trees.forEach((tree, treeIndex) => {
        // 获取每个特征在该树中的出现频率和位置
        const featureDepths = {};
        
        // 递归遍历树节点，记录特征使用情况
        const traverseTree = (node, depth = 0) => {
          if (!node || node.isLeaf) return;
          
          // 记录当前节点使用的特征及其深度
          const featureIdx = tree.featureSubset[node.varIndex];
          if (featureDepths[featureIdx] === undefined) {
            featureDepths[featureIdx] = [];
          }
          featureDepths[featureIdx].push(depth);
          
          // 继续遍历子节点
          traverseTree(node.left, depth + 1);
          traverseTree(node.right, depth + 1);
        };
        
        // 从根节点开始遍历
        traverseTree(tree.tree);
        
        // 根据深度计算特征重要性
        // 特征在越靠近根节点的位置出现越重要
        Object.entries(featureDepths).forEach(([featureIdx, depths]) => {
          // 将特征索引转换为数字
          const idx = parseInt(featureIdx);
          
          // 计算该特征在此树中的重要性
          // 公式: 深度的倒数之和，越靠近根节点贡献越大
          const depthBasedImportance = depths.reduce((sum, depth) => {
            return sum + 1 / (depth + 1); // +1避免除以零
          }, 0);
          
          importance[idx] += depthBasedImportance;
          featureCounts[idx] += 1;
        });
      });
      
      // 如果有些特征从未被使用，给予很小的非零值
      for (let i = 0; i < numFeatures; i++) {
        if (featureCounts[i] === 0) {
          importance[i] = 0.0001;
        } else {
          // 对于使用过的特征，除以使用次数获得平均重要性
          importance[i] /= featureCounts[i];
        }
      }
      
      // 归一化成百分比
      const totalImportance = importance.reduce((sum, val) => sum + val, 0);
      const normalizedImportance = importance.map(val => (val / totalImportance) * 100);
      
      return normalizedImportance;
    }
  }
  
  module.exports = RandomForestRegressor;