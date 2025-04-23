const fs = require('fs');
const path = require('path');
const Papa = require('papaparse');

// 构念定义
const constructDefinitions = {
  CPT: ['CPT1', 'CPT2', 'CPT3', 'CPT4'], // Compatibility
  RAD: ['RAD1', 'RAD2', 'RAD3', 'RAD4'], // Relative Advantage
  CPX: ['CPX1', 'CPX2', 'CPX3'], // Complexity
  PU: ['PU1', 'PU2', 'PU3', 'PU4'], // Perceived Usefulness
  PEOU: ['PEOU1', 'PEOU2', 'PEOU3', 'PEOU4'], // Perceived Ease of Use
  ATT: ['ATT1', 'ATT2', 'ATT3', 'ATT4'], // Attitude
  SN: ['SN1', 'SN2', 'SN3', 'SN4'], // Subjective Norms
  PBC: ['PBC1', 'PBC2'], // Perceived Behavioral Control
  BI: ['BI1', 'BI2'] // Behavioral Intention
};

// 加载CSV数据
function loadData(filePath) {
  const csvFile = fs.readFileSync(path.resolve(filePath), 'utf8');
  
  const parsedData = Papa.parse(csvFile, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: true
  });
  
  if (parsedData.errors.length > 0) {
    console.error("CSV解析错误:", parsedData.errors);
  }
  
  return parsedData.data;
}

// 计算复合得分
function calculateCompositeScores(data) {
  const compositeScores = {};
  
  Object.keys(constructDefinitions).forEach(construct => {
    compositeScores[construct] = [];
    
    data.forEach(row => {
      const variables = constructDefinitions[construct];
      const values = variables
        .map(variable => row[variable])
        .filter(value => value !== undefined && value !== null);
      
      if (values.length > 0) {
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        compositeScores[construct].push(mean);
      } else {
        compositeScores[construct].push(null);
      }
    });
  });
  
  return compositeScores;
}

// 准备机器学习数据集
function prepareMLDataset(compositeScores) {
  const mlDataset = [];
  const dataLength = compositeScores[Object.keys(compositeScores)[0]].length;
  
  for (let i = 0; i < dataLength; i++) {
    const validRow = Object.keys(compositeScores).every(
      construct => compositeScores[construct][i] !== null
    );
    
    if (validRow) {
      const row = {};
      Object.keys(compositeScores).forEach(construct => {
        row[construct] = compositeScores[construct][i];
      });
      mlDataset.push(row);
    }
  }
  
  return mlDataset;
}

// 提取特征和目标变量
function extractFeaturesAndTarget(mlDataset, featureNames, targetName) {
  const features = mlDataset.map(row => 
    featureNames.map(feature => row[feature])
  );
  
  const target = mlDataset.map(row => row[targetName]);
  
  return { features, target };
}

// 分割训练集和测试集
function trainTestSplit(features, target, testSize = 0.2) {
  const totalSamples = features.length;
  const testCount = Math.round(totalSamples * testSize);
  const trainCount = totalSamples - testCount;
  
  // 创建索引数组并打乱
  const indices = Array.from({ length: totalSamples }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  
  const trainIndices = indices.slice(0, trainCount);
  const testIndices = indices.slice(trainCount);
  
  return {
    trainFeatures: trainIndices.map(i => features[i]),
    trainTarget: trainIndices.map(i => target[i]),
    testFeatures: testIndices.map(i => features[i]),
    testTarget: testIndices.map(i => target[i])
  };
}

module.exports = {
  loadData,
  calculateCompositeScores,
  prepareMLDataset,
  extractFeaturesAndTarget,
  trainTestSplit,
  constructDefinitions
};