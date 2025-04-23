class DecisionTreeRegressor {
    constructor(maxDepth = 5) {
      this.maxDepth = maxDepth;
      this.tree = null;
    }
    
    findBestSplit(features, target) {
        const n = features.length;
        const numFeatures = features[0].length;
        
        let bestVarIndex = 0;
        let bestSplitValue = 0;
        let bestScore = -Infinity;
        
        // Try each feature as split variable
        for (let featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
          // Get all unique values for this feature
          const featureValues = features.map(f => f[featureIndex]);
          const uniqueValues = [...new Set(featureValues)].sort((a, b) => a - b);
          
          // If only one unique value, skip this feature
          if (uniqueValues.length <= 1) continue;
          
          // Try each value as a split point
          for (let i = 0; i < uniqueValues.length - 1; i++) {
            // Use midpoint between two consecutive unique values as split
            const splitValue = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
            
            // Split data based on this value
            const leftIndices = [];
            const rightIndices = [];
            
            for (let j = 0; j < n; j++) {
              if (features[j][featureIndex] <= splitValue) {
                leftIndices.push(j);
              } else {
                rightIndices.push(j);
              }
            }
            
            // Skip if one side is empty
            if (leftIndices.length === 0 || rightIndices.length === 0) continue;
            
            // Calculate score for this split (negative variance)
            const leftTarget = leftIndices.map(idx => target[idx]);
            const rightTarget = rightIndices.map(idx => target[idx]);
            
            const leftMean = leftTarget.reduce((sum, val) => sum + val, 0) / leftTarget.length;
            const rightMean = rightTarget.reduce((sum, val) => sum + val, 0) / rightTarget.length;
            
            // Calculate variance (MSE) for each side
            const leftVariance = leftTarget.reduce((sum, val) => sum + Math.pow(val - leftMean, 2), 0);
            const rightVariance = rightTarget.reduce((sum, val) => sum + Math.pow(val - rightMean, 2), 0);
            
            // Total score is negative of weighted variance (we want to maximize the score)
            const score = -(leftVariance + rightVariance);
            
            // Update best if this split is better
            if (score > bestScore) {
              bestScore = score;
              bestVarIndex = featureIndex;
              bestSplitValue = splitValue;
            }
          }
        }
        
        return { varIndex: bestVarIndex, value: bestSplitValue, score: bestScore };
      }
    
    buildTree(features, target, depth = 0) {
        const n = features.length;
    
        // Base cases: max depth reached or all targets are the same
        if (depth >= this.maxDepth || new Set(target).size <= 1) {
          // Create a leaf node
          const mean = target.reduce((sum, val) => sum + val, 0) / n;
          return { isLeaf: true, value: mean };
        }
        
        // Find the best split
        const split = this.findBestSplit(features, target);
        
        // If no good split found, create a leaf
        if (split.score === -Infinity) {
          const mean = target.reduce((sum, val) => sum + val, 0) / n;
          return { isLeaf: true, value: mean };
        }
        
        // Split the data
        const leftIndices = [];
        const rightIndices = [];
        
        for (let i = 0; i < n; i++) {
          if (features[i][split.varIndex] <= split.value) {
            leftIndices.push(i);
          } else {
            rightIndices.push(i);
          }
        }
        
        // Create features and targets for child nodes
        const leftFeatures = leftIndices.map(i => features[i]);
        const leftTarget = leftIndices.map(i => target[i]);
        
        const rightFeatures = rightIndices.map(i => features[i]);
        const rightTarget = rightIndices.map(i => target[i]);
        
        // Recursively build left and right subtrees
        const leftTree = this.buildTree(leftFeatures, leftTarget, depth + 1);
        const rightTree = this.buildTree(rightFeatures, rightTarget, depth + 1);
        
        // Return internal node
        return {
          isLeaf: false,
          varIndex: split.varIndex,
          value: split.value,
          left: leftTree,
          right: rightTree
        };
      }
    
    train(features, target) {
      this.tree = this.buildTree(features, target);
      return this;
    }
    
    predictSingle(x, node = this.tree) {
      if (node.isLeaf) {
        return node.value;
      }
      
      if (x[node.varIndex] <= node.value) {
        return this.predictSingle(x, node.left);
      } else {
        return this.predictSingle(x, node.right);
      }
    }
    
    predict(features) {
      return features.map(x => this.predictSingle(x));
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
    }
  
  module.exports = DecisionTreeRegressor;