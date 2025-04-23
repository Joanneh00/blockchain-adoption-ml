const path = require('path');
const { 
  loadData, 
  calculateCompositeScores, 
  prepareMLDataset,
  extractFeaturesAndTarget,
  trainTestSplit
} = require('./utils/dataLoader');
const LinearRegression = require('./models/linearRegression');
const DecisionTreeRegressor = require('./models/decisionTree');

// 主函数
async function main() {
  try {
    console.log("开始区块链采纳预测分析...");
    
    // 1. 加载数据
    const dataPath = path.join(__dirname, '../data/peerj-cs-11-2466-s001.csv');
    const data = loadData(dataPath);
    console.log(`成功加载 ${data.length} 条数据`);
    
    // 2. 计算复合得分
    const compositeScores = calculateCompositeScores(data);
    
    // 3. 准备ML数据集
    const mlDataset = prepareMLDataset(compositeScores);
    console.log(`准备了 ${mlDataset.length} 条有效数据点`);
    
    // 4. 定义特征和目标变量
    const primaryFeatures = ['CPT', 'RAD', 'CPX', 'SN', 'PBC'];
    const combinedFeatures = ['CPT', 'RAD', 'CPX', 'SN', 'PBC', 'PEOU', 'PU', 'ATT'];
    const targetVariable = 'BI';
    
    // 5. 提取基础模型的特征和目标
    const primaryModel = extractFeaturesAndTarget(mlDataset, primaryFeatures, targetVariable);
    
    // 6. 分割训练集和测试集
    const primarySplit = trainTestSplit(primaryModel.features, primaryModel.target, 0.2);
    console.log(`训练集大小: ${primarySplit.trainFeatures.length}`);
    console.log(`测试集大小: ${primarySplit.testFeatures.length}`);
    
    // 7. 训练和评估线性回归模型
    console.log("\n----- 线性回归模型 -----");
    const lr = new LinearRegression();
    lr.train(primarySplit.trainFeatures, primarySplit.trainTarget);
    
    const lrTrainScore = lr.score(primarySplit.trainFeatures, primarySplit.trainTarget);
    const lrTestScore = lr.score(primarySplit.testFeatures, primarySplit.testTarget);
    
    console.log("线性回归模型权重:", lr.weights);
    console.log("线性回归模型偏置:", lr.bias);
    console.log(`线性回归训练集 R²: ${lrTrainScore.toFixed(4)}`);
    console.log(`线性回归测试集 R²: ${lrTestScore.toFixed(4)}`);
    
    // 8. 特征重要性分析
    const featureImportance = lr.getFeatureImportance();
    console.log("\n特征重要性:");
    primaryFeatures.forEach((feature, index) => {
      console.log(`${feature}: ${featureImportance[index].toFixed(2)}% (权重: ${lr.weights[index].toFixed(4)})`);
    });
    
    // 9. 训练和评估决策树模型
    console.log("\n----- 决策树模型 -----");
    const dt = new DecisionTreeRegressor(3);
    dt.train(primarySplit.trainFeatures, primarySplit.trainTarget);
    
    const dtTrainScore = dt.score(primarySplit.trainFeatures, primarySplit.trainTarget);
    const dtTestScore = dt.score(primarySplit.testFeatures, primarySplit.testTarget);
    
    console.log(`决策树训练集 R²: ${dtTrainScore.toFixed(4)}`);
    console.log(`决策树测试集 R²: ${dtTestScore.toFixed(4)}`);
    
    // 10. 综合模型分析
    console.log("\n----- 综合模型 -----");
    const combinedModel = extractFeaturesAndTarget(mlDataset, combinedFeatures, targetVariable);
    const combinedSplit = trainTestSplit(combinedModel.features, combinedModel.target, 0.2);
    
    const lrCombined = new LinearRegression();
    lrCombined.train(combinedSplit.trainFeatures, combinedSplit.trainTarget);
    
    const lrCombinedTrainScore = lrCombined.score(combinedSplit.trainFeatures, combinedSplit.trainTarget);
    const lrCombinedTestScore = lrCombined.score(combinedSplit.testFeatures, combinedSplit.testTarget);
    
    console.log("综合模型权重:", lrCombined.weights);
    console.log("综合模型偏置:", lrCombined.bias);
    console.log(`综合模型训练集 R²: ${lrCombinedTrainScore.toFixed(4)}`);
    console.log(`综合模型测试集 R²: ${lrCombinedTestScore.toFixed(4)}`);
    
    // 11. 综合模型特征重要性
    const combinedFeatureImportance = lrCombined.getFeatureImportance();
    console.log("\n综合模型特征重要性:");
    combinedFeatures.forEach((feature, index) => {
      console.log(`${feature}: ${combinedFeatureImportance[index].toFixed(2)}% (权重: ${lrCombined.weights[index].toFixed(4)})`);
    });
    
    // 决策树综合模型分析
    console.log("\n----- 决策树综合模型 -----");
    const dtCombined = new DecisionTreeRegressor(3);  // 最大深度为3
    dtCombined.train(combinedSplit.trainFeatures, combinedSplit.trainTarget);

    const dtCombinedTrainScore = dtCombined.score(combinedSplit.trainFeatures, combinedSplit.trainTarget);
    const dtCombinedTestScore = dtCombined.score(combinedSplit.testFeatures, combinedSplit.testTarget);

    console.log(`决策树综合模型训练集 R²: ${dtCombinedTrainScore.toFixed(4)}`);
    console.log(`决策树综合模型测试集 R²: ${dtCombinedTestScore.toFixed(4)}`);

    // 分析决策树捕捉的非线性关系
    console.log("\n----- 决策树捕捉的非线性关系分析 -----");

    // 选择几个测试样本进行分析
    const sampleIndices = [0, 1, 2, 3, 4]; // 前5个测试样本

    // 比较线性和非线性模型的预测
    console.log("样本预测比较 (线性回归 vs 决策树):");
    sampleIndices.forEach(i => {
    const features = combinedSplit.testFeatures[i];
    const actual = combinedSplit.testTarget[i];
    const lrPredicted = lrCombined.predict(features);
    const dtPredicted = dtCombined.predictSingle(features);
    
    console.log(`\n样本 ${i+1}:`);
    console.log(`  实际值: ${actual.toFixed(2)}`);
    console.log(`  线性回归预测: ${lrPredicted.toFixed(2)}, 误差: ${Math.abs(actual - lrPredicted).toFixed(2)}`);
    console.log(`  决策树预测: ${dtPredicted.toFixed(2)}, 误差: ${Math.abs(actual - dtPredicted).toFixed(2)}`);
    
    // 简单分析哪个模型在这个样本上表现更好
    if (Math.abs(actual - lrPredicted) < Math.abs(actual - dtPredicted)) {
        console.log("  这个样本线性模型表现更好");
    } else if (Math.abs(actual - lrPredicted) > Math.abs(actual - dtPredicted)) {
        console.log("  这个样本非线性模型表现更好");
    } else {
        console.log("  两个模型表现相同");
    }
    
    // 打印样本特征值，看看是什么样的数据点
    console.log("  特征值:");
    combinedFeatures.forEach((feature, j) => {
        console.log(`    ${feature}: ${features[j].toFixed(2)}`);
    });
    });

    // 模型性能总结
    console.log("\n----- 模型性能比较 -----");
    console.log("基础线性回归模型 (5个特征) - 测试集 R²:", lrTestScore.toFixed(4));
    console.log("基础决策树模型 (5个特征) - 测试集 R²:", dtTestScore.toFixed(4));
    console.log("综合线性回归模型 (8个特征) - 测试集 R²:", lrCombinedTestScore.toFixed(4));
    console.log("综合决策树模型 (8个特征) - 测试集 R²:", dtCombinedTestScore.toFixed(4));

    // 总结分析结果
    console.log("\n----- 结论 -----");
    console.log("1. 特征重要性分析 (基于线性回归):");
    // 找出最重要的三个特征
    const topFeatures = [...combinedFeatures]
    .map((feature, index) => ({ feature, importance: combinedFeatureImportance[index], weight: lrCombined.weights[index] }))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 3);

    topFeatures.forEach((item, index) => {
    const direction = item.weight >= 0 ? "正向" : "负向";
    console.log(`   ${index+1}. ${item.feature}: ${item.importance.toFixed(2)}% (${direction}影响)`);
    });

    console.log("\n2. 非线性关系分析 (基于决策树):");
    // 比较线性和非线性模型的整体表现
    if (dtCombinedTestScore > lrCombinedTestScore) {
    console.log("   决策树模型表现更好，表明数据中可能存在重要的非线性关系");
    } else {
    console.log("   线性回归模型表现更好，表明线性关系可能足以解释大部分变异");
    }


    console.log("\n分析完成!");
    
  } catch (error) {
    console.error("分析过程中出错:", error);
  }
}

// 执行主函数
main();