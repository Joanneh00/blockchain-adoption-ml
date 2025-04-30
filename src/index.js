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

    // Fine Tune Decision Tree
    console.log("\nTuning Decision Tree Model...");
    const maxDepths = [2, 3, 4, 5];
    let bestDtParams = null;
    let bestDtScore = -Infinity;

    for (const depth of maxDepths) {
      // 对当前参数进行5折交叉验证
      const scores = crossValidate(
        primaryModel.features, 
        primaryModel.target, 
        5, 
        () => {
          const model = new DecisionTreeRegressor(depth);
          return model;
        }
      );
      
      // 计算平均分数
      const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
      const stdDev = Math.sqrt(
        scores.reduce((sum, s) => sum + Math.pow(s - avgScore, 2), 0) / scores.length
      );
      
      console.log(`  Max Depth: ${depth}, Avg R²: ${avgScore.toFixed(4)}, StdDev: ${stdDev.toFixed(4)}`);
      
      // 更新最佳参数
      if (avgScore > bestDtScore) {
        bestDtScore = avgScore;
        bestDtParams = { maxDepth: depth };
      }
    }

    console.log(`\nBest Decision Tree Parameters: Max Depth = ${bestDtParams.maxDepth}`);
    console.log(`Best Cross-Validation R²: ${bestDtScore.toFixed(4)}`);

    // Fine Tune Random Forest
    console.log("\nTuning Random Forest Model...");
    const numTreesOptions = [10, 20, 30];
    const rfMaxDepths = [2, 3, 4];
    let bestRfParams = null;
    let bestRfScore = -Infinity;

    for (const numTrees of numTreesOptions) {
      for (const depth of rfMaxDepths) {
        // 由于随机森林计算较为耗时，使用3折交叉验证
        const scores = crossValidate(
          primaryModel.features, 
          primaryModel.target, 
          3, 
          () => {
            const model = new RandomForestRegressor(numTrees, depth);
            return model;
          }
        );
        
        // 计算平均分数
        const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
        const stdDev = Math.sqrt(
          scores.reduce((sum, s) => sum + Math.pow(s - avgScore, 2), 0) / scores.length
        );
        
        console.log(`  Trees: ${numTrees}, Max Depth: ${depth}, Avg R²: ${avgScore.toFixed(4)}, StdDev: ${stdDev.toFixed(4)}`);
        
        // 更新最佳参数
        if (avgScore > bestRfScore) {
          bestRfScore = avgScore;
          bestRfParams = { numTrees, maxDepth: depth };
        }
      }
    }

    console.log(`\nBest Random Forest Parameters: Trees = ${bestRfParams.numTrees}, Max Depth = ${bestRfParams.maxDepth}`);
    console.log(`Best Cross-Validation R²: ${bestRfScore.toFixed(4)}`);

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
    
    // /*-------------------------------------------------
    //   MODEL 2: DECISION TREE ANALYSIS 
    // -------------------------------------------------*/

    console.log("\n----- DECISION TREE MODEL -----");
    // 使用最佳参数训练决策树模型
    const dt = new DecisionTreeRegressor(bestDtParams.maxDepth);
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
    // use best parameter for train 
    const rf = new RandomForestRegressor(bestRfParams.numTrees, bestRfParams.maxDepth);
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
    const dtCombined = new DecisionTreeRegressor(bestDtParams.maxDepth);
    dtCombined.train(combinedSplit.trainFeatures, combinedSplit.trainTarget);
    
    const dtCombinedTrainScore = dtCombined.score(combinedSplit.trainFeatures, combinedSplit.trainTarget);
    const dtCombinedTestScore = dtCombined.score(combinedSplit.testFeatures, combinedSplit.testTarget);
    
    console.log(`Comprehensive Decision Tree Training R²: ${dtCombinedTrainScore.toFixed(4)}`);
    console.log(`Comprehensive Decision Tree Test R²: ${dtCombinedTestScore.toFixed(4)}`);
    
    // 3. Comprehensive Random Forest Model
    console.log("\n3. Comprehensive Random Forest Model:");
    const rfCombined = new RandomForestRegressor(bestRfParams.numTrees, bestRfParams.maxDepth);
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
     FEATURE INTERACTION ANALYSIS
    -------------------------------------------------*/

    // 特征交互分析
    function improvedFeatureInteractionAnalysis(rfModel, features, testFeatures, testTarget) {
      console.log("\n----- FEATURE INTERACTION ANALYSIS -----");
      
      // 创建特征对
      const featurePairs = [];
      for (let i = 0; i < features.length; i++) {
        for (let j = i+1; j < features.length; j++) {
          featurePairs.push([i, j]);
        }
      }
      
      console.log("Potential Feature Interactions:");
      
      // 分析前5个交互
      featurePairs.slice(0, 27).forEach(pair => {
        const [i, j] = pair;
        
        console.log(`\n${features[i]} and ${features[j]} interaction:`);
        
        // 创建基准样本
        const baselineSample = Array(features.length).fill(3); // 中等值
        
        // 创建四种组合的样本
        const lowLowSample = [...baselineSample];
        lowLowSample[i] = 1.5; // 低值
        lowLowSample[j] = 1.5; // 低值
        
        const highHighSample = [...baselineSample];
        highHighSample[i] = 4.5; // 高值
        highHighSample[j] = 4.5; // 高值
        
        // 使用随机森林预测这些样本
        const lowLowPred = rfModel.predictSingle(lowLowSample);
        const highHighPred = rfModel.predictSingle(highHighSample);
        
        // 根据真实预测分析交互
        console.log(`  - When both features are low, prediction = ${lowLowPred.toFixed(2)}`);
        console.log(`  - When both features are high, prediction = ${highHighPred.toFixed(2)}`);
        
        // 计算差异来确定交互存在与否
        const difference = highHighPred - lowLowPred;
        const interactionExists = Math.abs(difference) > 0.5;
        
        console.log(`  - This suggests a significant interaction between these features ${interactionExists ? 'exists' : 'does not exist'}`);
      });
    }

     /*-------------------------------------------------
     Threshold Analysis ANALYSIS
    -------------------------------------------------*/

    // 阈值效应分析
    function improvedThresholdAnalysis(rfModel, features, testFeatures, testTarget) {
      console.log("\nThreshold Effect Analysis:");
      
      // 分析前3个特征
      features.slice(0, 27).forEach((feature, idx) => {
        console.log(`\n${feature} threshold effects:`);
        
        // 创建基准样本
        const baselineSample = Array(features.length).fill(3);
        
        // 测试不同阈值
        const thresholds = [1.5, 2.5, 3.5, 4.5];
        const predictions = [];
        
        // 使用模型预测不同阈值下的结果
        thresholds.forEach(threshold => {
          const testSample = [...baselineSample];
          testSample[idx] = threshold;
          const prediction = rfModel.predictSingle(testSample);
          predictions.push({ threshold, prediction });
        });
        
        // 分析预测趋势，查找可能的阈值点
        let hasThreshold = false;
        for (let i = 1; i < predictions.length; i++) {
          const change = predictions[i].prediction - predictions[i-1].prediction;
          if (Math.abs(change) > 0.5) {
            hasThreshold = true;
            console.log(`  - Significant change detected at threshold ${predictions[i-1].threshold}`);
          }
        }
        
        // 输出低值和高值的预测
        console.log(`  - When ${feature} < 2.5, predictions tend to ${predictions[0].prediction < predictions[1].prediction ? 'decrease' : 'increase'}`);
        console.log(`  - When ${feature} > 3.5, predictions tend to ${predictions[2].prediction < predictions[3].prediction ? 'decrease' : 'increase'}`);
        console.log(`  - This suggests that a threshold effect for ${feature} ${hasThreshold ? 'might exist' : 'might not exist'}`);
      });
    }

    improvedFeatureInteractionAnalysis(
      rfCombined,
      combinedFeatures,
      combinedSplit.testFeatures,
      combinedSplit.testTarget
    );
    
    // 3. 阈值效应分析 (使用改进的函数)
    improvedThresholdAnalysis(
      rfCombined,
      combinedFeatures,
      combinedSplit.testFeatures,
      combinedSplit.testTarget
    );
    
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
    
    /*-----------------------------------------------
    Culculate MSE
    ------------------------------------------------*/
    function calculateMSE(actual, predicted) {
      if (!Array.isArray(actual) || !Array.isArray(predicted)) {
        console.error('Input must be arrays');
        console.log('Actual:', actual);
        console.log('Predicted:', predicted);
        throw new Error('Input must be arrays');
      }
      
      if (actual.length !== predicted.length) {
        console.error(`Arrays length mismatch: actual ${actual.length}, predicted ${predicted.length}`);
        // 你可以选择使用较短的长度来避免错误
        const minLength = Math.min(actual.length, predicted.length);
        actual = actual.slice(0, minLength);
        predicted = predicted.slice(0, minLength);
        console.warn(`Using first ${minLength} elements of each array instead`);
      }
      
      let sumSquaredError = 0;
      for (let i = 0; i < actual.length; i++) {
        sumSquaredError += Math.pow(actual[i] - predicted[i], 2);
      }
      
      return sumSquaredError / actual.length;
    }
    
    /*-----------------------------------------------
    Culculate RMSE
    ------------------------------------------------*/
    function calculateRMSE(actual, predicted) {
      return Math.sqrt(calculateMSE(actual, predicted));
    }

    /*-----------------------------------------------
    Culculate MSE & RMSE for LR, DT, RF
    ------------------------------------------------*/
    const lrPredictions = lr.predictBatch(primarySplit.testFeatures);
    const lrTestMSE = calculateMSE(primarySplit.testTarget, lrPredictions);
    const lrTestRMSE = calculateRMSE(primarySplit.testTarget, lrPredictions);

    console.log(`\nLinear Regression Test MSE: ${lrTestMSE.toFixed(4)}`);
    console.log(`Linear Regression Test RMSE: ${lrTestRMSE.toFixed(4)}`);

    DecisionTreeRegressor.prototype.predictBatch = function(features) {
      return features.map(x => this.predictSingle(x));
    };

    RandomForestRegressor.prototype.predictBatch = function(features) {
      return features.map(x => this.predictSingle(x));
    };

    // 决策树模型的MSE和RMSE
    const dtPredictions = dt.predictBatch(primarySplit.testFeatures);
    const dtTestMSE = calculateMSE(primarySplit.testTarget, dtPredictions);
    const dtTestRMSE = calculateRMSE(primarySplit.testTarget, dtPredictions);
    console.log(`\nDecision Tree Test MSE: ${dtTestMSE.toFixed(4)}`);
    console.log(`Decision Tree Test RMSE: ${dtTestRMSE.toFixed(4)}`);

    // 随机森林模型的MSE和RMSE
    const rfPredictions = rf.predictBatch(primarySplit.testFeatures);
    const rfTestMSE = calculateMSE(primarySplit.testTarget, rfPredictions);
    const rfTestRMSE = calculateRMSE(primarySplit.testTarget, rfPredictions);
    console.log(`\nRandom Forest Test MSE: ${rfTestMSE.toFixed(4)}`);
    console.log(`Random Forest Test RMSE: ${rfTestRMSE.toFixed(4)}`);

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