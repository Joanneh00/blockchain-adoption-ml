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
        const randomIndex = Math.floor(Math.random() * n);
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
        const j = Math.floor(Math.random() * (i + 1));
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
    
    // Feature importance for random forest is calculated as the mean decrease in impurity
    getFeatureImportance(numFeatures) {
      if (this.trees.length === 0) {
        throw new Error("Model not trained yet");
      }
      
      // Initialize importance array with zeros
      const importance = Array(numFeatures).fill(0);
      
      // Count how many times each feature is used across all trees
      const featureCounts = Array(numFeatures).fill(0);
      
      // For each tree in the forest
      this.trees.forEach(tree => {
        // For each feature in the tree's subset
        tree.featureSubset.forEach((featureIdx, i) => {
          // Add feature importance from this tree
          // We'll use a simple approximation here - features used higher in the tree are more important
          // A more accurate approach would require tracking each node's impurity decrease
          importance[featureIdx] += 1 / (tree.featureSubset.length);
          featureCounts[featureIdx] += 1;
        });
      });
      
      // Average the importance by the number of times the feature was used
      for (let i = 0; i < numFeatures; i++) {
        if (featureCounts[i] > 0) {
          importance[i] /= featureCounts[i];
        }
      }
      
      // Normalize to sum to 100%
      const totalImportance = importance.reduce((sum, val) => sum + val, 0);
      const normalizedImportance = importance.map(val => (val / totalImportance) * 100);
      
      return normalizedImportance;
    }
  }
  
  module.exports = RandomForestRegressor;