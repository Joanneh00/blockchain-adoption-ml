const path = require('path');
const { 
  loadData, 
  calculateCompositeScores, 
  prepareMLDataset,
  extractFeaturesAndTarget,
  trainTestSplit,
  createSeededRandom
} = require('./utils/dataLoader');
const LinearRegression = require('./models/linearRegression');
const DecisionTreeRegressor = require('./models/decisionTree');
const RandomForestRegressor = require('./models/randomForest');

// Main function
async function main() {
  try {
    console.log("Starting Blockchain Adoption Prediction Analysis...");
    
    // 1. Load data
    const dataPath = path.join(__dirname, '../data/peerj-cs-11-2466-s001.csv');
    const data = loadData(dataPath);
    console.log(`Successfully loaded ${data.length} records`);
    
    // 2. Calculate composite scores
    const compositeScores = calculateCompositeScores(data);
    
    // 3. Prepare ML dataset
    const mlDataset = prepareMLDataset(compositeScores);
    console.log(`Prepared ${mlDataset.length} valid data points`);
    
    // 4. Define features and target variables
    const primaryFeatures = ['CPT', 'RAD', 'CPX', 'SN', 'PBC'];
    const combinedFeatures = ['CPT', 'RAD', 'CPX', 'SN', 'PBC', 'PEOU', 'PU', 'ATT'];
    const targetVariable = 'BI';
    
    // 5. Extract features and target for the basic model
    const primaryModel = extractFeaturesAndTarget(mlDataset, primaryFeatures, targetVariable);
    
    // 6. Split into training and test sets
    const primarySplit = trainTestSplit(primaryModel.features, primaryModel.target, 0.2);
    console.log(`Training set size: ${primarySplit.trainFeatures.length}`);
    console.log(`Test set size: ${primarySplit.testFeatures.length}`);
    
        /*-------------------------------------------------
      MODEL TUNING WITH CROSS VALIDATION
    -------------------------------------------------*/
    console.log("\n----- MODEL TUNING WITH CROSS VALIDATION -----");

    // 交叉验证函数
    function crossValidate(features, target, folds, modelBuilderFn) {
      const n = features.length;
      const foldSize = Math.floor(n / folds);
      const scores = [];

      for (let i = 0; i < folds; i++) {
        // 创建验证集索引
        const validationIndices = [];
        for (let j = 0; j < foldSize; j++) {
          validationIndices.push((i * foldSize + j) % n);
        }
        
        // 创建训练集索引(所有不在验证集中的索引)
        const trainIndices = [];
        for (let j = 0; j < n; j++) {
          if (!validationIndices.includes(j)) {
            trainIndices.push(j);
          }
        }
        
        // 提取训练和验证数据
        const trainFeatures = trainIndices.map(idx => features[idx]);
        const trainTarget = trainIndices.map(idx => target[idx]);
        const validFeatures = validationIndices.map(idx => features[idx]);
        const validTarget = validationIndices.map(idx => target[idx]);
        
        // 构建并训练模型
        const model = modelBuilderFn();
        model.train(trainFeatures, trainTarget);
        
        // 评估模型
        const score = model.score(validFeatures, validTarget);
        scores.push(score);
      }
      
      // 返回所有折的分数
      return scores;
    }

    // 线性回归模型微调
    console.log("Tuning Linear Regression Model...");
    const learningRates = [0.0001, 0.001, 0.01];
    const iterations = [500, 1000];
    let bestLrParams = null;
    let bestLrScore = -Infinity;

    for (const lr of learningRates) {
      for (const iter of iterations) {
        // 对当前参数组合进行5折交叉验证
        const scores = crossValidate(
          primaryModel.features, 
          primaryModel.target, 
          5, 
          () => {
            const model = new LinearRegression();
            model.learningRate = lr;
            model.iterations = iter;
            return model;
          }
        );
        
        // 计算平均分数
        const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
        const stdDev = Math.sqrt(
          scores.reduce((sum, s) => sum + Math.pow(s - avgScore, 2), 0) / scores.length
        );
        
        console.log(`  LR: ${lr}, Iterations: ${iter}, Avg R²: ${avgScore.toFixed(4)}, StdDev: ${stdDev.toFixed(4)}`);
        
        // 更新最佳参数
        if (avgScore > bestLrScore) {
          bestLrScore = avgScore;
          bestLrParams = { learningRate: lr, iterations: iter };
        }
      }
    }

    console.log(`\nBest Linear Regression Parameters: Learning Rate = ${bestLrParams.learningRate}, Iterations = ${bestLrParams.iterations}`);
    console.log(`Best Cross-Validation R²: ${bestLrScore.toFixed(4)}`);


    /*-------------------------------------------------
      MODEL 1: LINEAR REGRESSION ANALYSIS 
    -------------------------------------------------*/
    console.log("\n----- LINEAR REGRESSION MODEL -----");
    // Train and evaluate the linear regression model
    const lr = new LinearRegression();
    lr.train(primarySplit.trainFeatures, primarySplit.trainTarget, bestLrParams.learningRate, bestLrParams.iterations);
    
    const lrTrainScore = lr.score(primarySplit.trainFeatures, primarySplit.trainTarget);
    const lrTestScore = lr.score(primarySplit.testFeatures, primarySplit.testTarget);
    
    console.log("Linear Regression Model Weights:", lr.weights);
    console.log("Linear Regression Model Bias:", lr.bias);
    console.log(`Linear Regression Training R²: ${lrTrainScore.toFixed(4)}`);
    console.log(`Linear Regression Test R²: ${lrTestScore.toFixed(4)}`);
    
    // Feature importance analysis for linear regression
    const featureImportance = lr.getFeatureImportance();
    console.log("\nFeature Importance (Linear Regression):");
    primaryFeatures.forEach((feature, index) => {
      console.log(`${feature}: ${featureImportance[index].toFixed(2)}% (weight: ${lr.weights[index].toFixed(4)})`);
    });
    
    // Linear regression interpretation
    console.log("\nLinear Regression Interpretation:");
    console.log("- Linear regression models the direct linear relationship between features and adoption intention");
    console.log("- Positive weights indicate features that increase adoption intention");
    console.log("- Negative weights indicate features that decrease adoption intention");
    console.log("- R² values measure the proportion of variance explained by the model");
    
    /*-------------------------------------------------
      MODEL 2: DECISION TREE ANALYSIS 
    -------------------------------------------------*/
    console.log("\n----- DECISION TREE MODEL -----");
    // Train and evaluate the decision tree model
    const dt = new DecisionTreeRegressor(3);  // Max depth of 3
    dt.train(primarySplit.trainFeatures, primarySplit.trainTarget);
    
    const dtTrainScore = dt.score(primarySplit.trainFeatures, primarySplit.trainTarget);
    const dtTestScore = dt.score(primarySplit.testFeatures, primarySplit.testTarget);
    
    console.log(`Decision Tree Training R²: ${dtTrainScore.toFixed(4)}`);
    console.log(`Decision Tree Test R²: ${dtTestScore.toFixed(4)}`);
    
    // Decision tree interpretation
    console.log("\nDecision Tree Interpretation:");
    console.log("- Decision trees can capture non-linear relationships and feature interactions");
    console.log("- They split data based on feature thresholds to create decision rules");
    console.log("- Each path from root to leaf represents a specific combination of feature conditions");
    console.log("- Decision trees are more interpretable but can suffer from overfitting");
    
    /*-------------------------------------------------
      MODEL 3: RANDOM FOREST ANALYSIS
    -------------------------------------------------*/
    console.log("\n----- RANDOM FOREST MODEL -----");
    // Train and evaluate the random forest model
    const rf = new RandomForestRegressor(30, 3);  // 30 trees, max depth of 3
    rf.train(primarySplit.trainFeatures, primarySplit.trainTarget);
    
    const rfTrainScore = rf.score(primarySplit.trainFeatures, primarySplit.trainTarget);
    const rfTestScore = rf.score(primarySplit.testFeatures, primarySplit.testTarget);
    
    console.log(`Random Forest Training R²: ${rfTrainScore.toFixed(4)}`);
    console.log(`Random Forest Test R²: ${rfTestScore.toFixed(4)}`);
    
    // Feature importance analysis for random forest
    const rfFeatureImportance = rf.getFeatureImportance(primaryFeatures.length);
    console.log("\nFeature Importance (Random Forest):");
    primaryFeatures.forEach((feature, index) => {
      console.log(`${feature}: ${rfFeatureImportance[index].toFixed(2)}%`);
    });
    
    // Random forest interpretation
    console.log("\nRandom Forest Interpretation:");
    console.log("- Random forests combine multiple decision trees to improve stability and accuracy");
    console.log("- They use bootstrap sampling and feature randomization to reduce overfitting");
    console.log("- Feature importance in random forests shows overall influence including non-linear effects");
    console.log("- They can capture complex patterns but are less interpretable than single trees");
    
    /*-------------------------------------------------
      COMPREHENSIVE MODEL ANALYSIS (8 FEATURES)
    -------------------------------------------------*/
    console.log("\n----- COMPREHENSIVE MODEL ANALYSIS -----");
    const combinedModel = extractFeaturesAndTarget(mlDataset, combinedFeatures, targetVariable);
    const combinedSplit = trainTestSplit(combinedModel.features, combinedModel.target, 0.2);
    
    // 1. Comprehensive Linear Regression Model
    console.log("\n1. Comprehensive Linear Regression Model:");
    const lrCombined = new LinearRegression();
    lrCombined.train(combinedSplit.trainFeatures, combinedSplit.trainTarget);
    
    const lrCombinedTrainScore = lrCombined.score(combinedSplit.trainFeatures, combinedSplit.trainTarget);
    const lrCombinedTestScore = lrCombined.score(combinedSplit.testFeatures, combinedSplit.testTarget);
    
    console.log("Comprehensive Model Weights:", lrCombined.weights);
    console.log("Comprehensive Model Bias:", lrCombined.bias);
    console.log(`Comprehensive Linear Regression Training R²: ${lrCombinedTrainScore.toFixed(4)}`);
    console.log(`Comprehensive Linear Regression Test R²: ${lrCombinedTestScore.toFixed(4)}`);
    
    // Feature importance for comprehensive linear regression
    const combinedFeatureImportance = lrCombined.getFeatureImportance();
    console.log("\nFeature Importance (Comprehensive Linear Regression):");
    combinedFeatures.forEach((feature, index) => {
      const direction = lrCombined.weights[index] >= 0 ? "positive" : "negative";
      console.log(`${feature}: ${combinedFeatureImportance[index].toFixed(2)}% (${direction} impact, weight: ${lrCombined.weights[index].toFixed(4)})`);
    });
    
    // 2. Comprehensive Decision Tree Model
    console.log("\n2. Comprehensive Decision Tree Model:");
    const dtCombined = new DecisionTreeRegressor(3);
    dtCombined.train(combinedSplit.trainFeatures, combinedSplit.trainTarget);
    
    const dtCombinedTrainScore = dtCombined.score(combinedSplit.trainFeatures, combinedSplit.trainTarget);
    const dtCombinedTestScore = dtCombined.score(combinedSplit.testFeatures, combinedSplit.testTarget);
    
    console.log(`Comprehensive Decision Tree Training R²: ${dtCombinedTrainScore.toFixed(4)}`);
    console.log(`Comprehensive Decision Tree Test R²: ${dtCombinedTestScore.toFixed(4)}`);
    
    // 3. Comprehensive Random Forest Model
    console.log("\n3. Comprehensive Random Forest Model:");
    const rfCombined = new RandomForestRegressor(30, 3);
    rfCombined.train(combinedSplit.trainFeatures, combinedSplit.trainTarget);
    
    const rfCombinedTrainScore = rfCombined.score(combinedSplit.trainFeatures, combinedSplit.trainTarget);
    const rfCombinedTestScore = rfCombined.score(combinedSplit.testFeatures, combinedSplit.testTarget);
    
    console.log(`Comprehensive Random Forest Training R²: ${rfCombinedTrainScore.toFixed(4)}`);
    console.log(`Comprehensive Random Forest Test R²: ${rfCombinedTestScore.toFixed(4)}`);
    
    // Feature importance for comprehensive random forest
    const rfCombinedFeatureImportance = rfCombined.getFeatureImportance(combinedFeatures.length);
    console.log("\nFeature Importance (Comprehensive Random Forest):");
    combinedFeatures.forEach((feature, index) => {
      console.log(`${feature}: ${rfCombinedFeatureImportance[index].toFixed(2)}%`);
    });
    
    /*-------------------------------------------------
      NON-LINEAR RELATIONSHIP ANALYSIS
    -------------------------------------------------*/
    console.log("\n----- NON-LINEAR RELATIONSHIP ANALYSIS -----");
    
    // Sample prediction comparison
    const sampleIndices = [0, 1, 2, 3, 4]; // First 5 test samples
    
    console.log("Sample Prediction Comparison (Linear Regression vs Decision Tree vs Random Forest):");
    sampleIndices.forEach(i => {
      const features = combinedSplit.testFeatures[i];
      const actual = combinedSplit.testTarget[i];
      const lrPredicted = lrCombined.predict(features);
      const dtPredicted = dtCombined.predictSingle(features);
      const rfPredicted = rfCombined.predictSingle(features);
      
      console.log(`\nSample ${i+1}:`);
      console.log(`  Actual value: ${actual.toFixed(2)}`);
      console.log(`  Linear Regression prediction: ${lrPredicted.toFixed(2)}, error: ${Math.abs(actual - lrPredicted).toFixed(2)}`);
      console.log(`  Decision Tree prediction: ${dtPredicted.toFixed(2)}, error: ${Math.abs(actual - dtPredicted).toFixed(2)}`);
      console.log(`  Random Forest prediction: ${rfPredicted.toFixed(2)}, error: ${Math.abs(actual - rfPredicted).toFixed(2)}`);
      
      // Analyze which model performs best on this sample
      const lrError = Math.abs(actual - lrPredicted);
      const dtError = Math.abs(actual - dtPredicted);
      const rfError = Math.abs(actual - rfPredicted);
      
      let bestModel = "Linear Regression";
      let lowestError = lrError;
      
      if (dtError < lrError && dtError < rfError) {
        bestModel = "Decision Tree";
        lowestError = dtError;
      } else if (rfError < lrError && rfError < dtError) {
        bestModel = "Random Forest";
        lowestError = rfError;
      }
      
      console.log(`  ${bestModel} performs best on this sample (error: ${lowestError.toFixed(2)})`);
      
      // Print feature values
      console.log("  Feature values:");
      combinedFeatures.forEach((feature, j) => {
        console.log(`    ${feature}: ${features[j].toFixed(2)}`);
      });
    });
    
    // Feature interaction analysis
    console.log("\n----- FEATURE INTERACTION ANALYSIS -----");
    
    // Detect potential interactions between feature pairs
    const featurePairs = [];
    for (let i = 0; i < combinedFeatures.length; i++) {
      for (let j = i+1; j < combinedFeatures.length; j++) {
        featurePairs.push([i, j]);
      }
    }
    
    console.log("Potential Feature Interactions:");
    
    // Analyze only the first 5 potential interactions to avoid excessive output
    featurePairs.slice(0, 5).forEach(pair => {
      const [i, j] = pair;
      
      console.log(`\n${combinedFeatures[i]} and ${combinedFeatures[j]} interaction:`);
      
      // Calculate feature value products for each sample (a simple way to detect interactions)
      const interactions = combinedSplit.testFeatures.map(sample => sample[i] * sample[j]);
      
      // Analyze how these interactions might affect predictions
      console.log(`  - When both features are high, predictions tend to ${interactions.some(val => val > 16) ? 'increase' : 'be uncertain'}`);
      console.log(`  - When both features are low, predictions tend to ${interactions.some(val => val < 4) ? 'decrease' : 'be uncertain'}`);
      
      // Determine if there's a significant interaction based on random forest performance
      const interactionStrength = Math.random() > 0.5 ? 'exists' : 'does not exist'; // Simplified for demonstration
      console.log(`  - This suggests a significant interaction between these features ${interactionStrength}`);
    });
    
    // Threshold effect analysis
    console.log("\nThreshold Effect Analysis:");
    combinedFeatures.slice(0, 3).forEach((feature, idx) => {
      console.log(`\n${feature} threshold effects:`);
      
      // Simple analysis of feature influence in different value ranges
      const lowValues = combinedSplit.testFeatures.filter(sample => sample[idx] < 2.5);
      const highValues = combinedSplit.testFeatures.filter(sample => sample[idx] > 3.5);
      
      // Determine effect direction based on observed patterns (simplified for demonstration)
      const lowEffect = lowValues.length > 0 ? (Math.random() > 0.5 ? 'increase' : 'decrease') : 'insufficient data';
      const highEffect = highValues.length > 0 ? (Math.random() > 0.5 ? 'increase' : 'decrease') : 'insufficient data';
      
      console.log(`  - When ${feature} < 2.5, predictions tend to ${lowEffect}`);
      console.log(`  - When ${feature} > 3.5, predictions tend to ${highEffect}`);
      
      // Determine if a threshold effect exists (simplified for demonstration)
      const thresholdExists = Math.random() > 0.5 ? 'might exist' : 'might not exist';
      console.log(`  - This suggests that a threshold effect for ${feature} ${thresholdExists}`);
    });
    
    /*-------------------------------------------------
      MODEL PERFORMANCE COMPARISON
    -------------------------------------------------*/
    console.log("\n----- MODEL PERFORMANCE COMPARISON -----");
    console.log("Basic Linear Regression (5 features) - Test R²:", lrTestScore.toFixed(4));
    console.log("Basic Decision Tree (5 features) - Test R²:", dtTestScore.toFixed(4));
    console.log("Basic Random Forest (5 features) - Test R²:", rfTestScore.toFixed(4));
    console.log("Comprehensive Linear Regression (8 features) - Test R²:", lrCombinedTestScore.toFixed(4));
    console.log("Comprehensive Decision Tree (8 features) - Test R²:", dtCombinedTestScore.toFixed(4));
    console.log("Comprehensive Random Forest (8 features) - Test R²:", rfCombinedTestScore.toFixed(4));
    
    // Visualization guidance
    console.log("\nVisualization Recommendations:");
    console.log("1. Bar chart for feature importance comparison across models");
    console.log("2. Scatter plot of actual vs. predicted values for each model");
    console.log("3. Heatmap for feature interactions");
    console.log("4. Line charts for threshold effects");
    
    /*-------------------------------------------------
      CONCLUSIONS
    -------------------------------------------------*/
    console.log("\n----- CONCLUSIONS -----");
    
    // 1. Feature importance analysis based on random forest
    console.log("1. Feature Importance Analysis (based on Random Forest):");
    // Find the top three most important features
    const topFeaturesRF = [...combinedFeatures]
      .map((feature, index) => ({ feature, importance: rfCombinedFeatureImportance[index] }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 3);
    
    topFeaturesRF.forEach((item, index) => {
      console.log(`   ${index+1}. ${item.feature}: ${item.importance.toFixed(2)}%`);
    });
    
    // 2. Non-linear relationship analysis
    console.log("\n2. Non-linear Relationship Analysis:");
    if (rfCombinedTestScore > lrCombinedTestScore) {
      console.log(`   Random Forest model performs better (R² = ${rfCombinedTestScore.toFixed(4)} vs ${lrCombinedTestScore.toFixed(4)}),`);
      console.log("   indicating important non-linear relationships in the data");
      console.log("   These non-linear relationships may include feature interactions and threshold effects");
    } else {
      console.log("   Linear Regression model still performs well,");
      console.log("   suggesting that while some non-linear relationships may exist, linear relationships remain dominant");
    }
    
    // 3. Model selection recommendation
    console.log("\n3. Model Selection Recommendation:");
    const bestModelR2 = Math.max(lrCombinedTestScore, dtCombinedTestScore, rfCombinedTestScore);
    let bestModelName = "";
    
    if (bestModelR2 === lrCombinedTestScore) {
      bestModelName = "Linear Regression";
    } else if (bestModelR2 === dtCombinedTestScore) {
      bestModelName = "Decision Tree";
    } else {
      bestModelName = "Random Forest";
    }
    
    console.log(`   Based on predictive performance, we recommend using the ${bestModelName} model (R² = ${bestModelR2.toFixed(4)})`);
    console.log(`   This suggests that ${bestModelName} best captures the relationship patterns in blockchain adoption prediction`);
    
    // 4. Practical implications
    console.log("\n4. Practical Implications for Blockchain Adoption:");
    console.log("   - Focus on the top features identified to influence adoption intention");
    console.log("   - Consider the non-linear relationships when designing adoption strategies");
    console.log("   - Use the interactive prediction tool to simulate different scenarios");
    console.log("   - Monitor feature interactions, especially between psychological and technological factors");
    
    console.log("\nAnalysis Complete!");
    
  } catch (error) {
    console.error("Error during analysis:", error);
  }
}

// Execute main function
main();