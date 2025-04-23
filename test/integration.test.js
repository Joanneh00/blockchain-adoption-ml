const path = require('path');
const { 
  loadData, 
  calculateCompositeScores, 
  prepareMLDataset, 
  extractFeaturesAndTarget, 
  trainTestSplit 
} = require('../src/utils/dataLoader.js'); // 修改为您模块的实际路径

describe('Integration Tests with Real CSV File', () => {
  // 指定测试数据文件的路径
  //const csvFilePath = path.join(__dirname, 'fixtures', 'test_data.csv');
  // 如果您想使用您原有的数据文件，可以这样:
   const csvFilePath = path.join(__dirname, '..', 'data', 'peerj-cs-11-2466-s001.csv');
  
  test('should successfully load and process real CSV data', () => {
    // 1. 加载CSV数据
    const data = loadData(csvFilePath);
    
    // 确认数据已正确加载
    expect(data).toBeDefined();
    expect(Array.isArray(data)).toBe(true);
    expect(data.length).toBeGreaterThan(0);
    
    // 检查数据结构
    console.log('First row of data:', data[0]);
    
    // 2. 计算复合得分
    const scores = calculateCompositeScores(data);
    
    // 检查得分是否已计算
    expect(scores).toBeDefined();
    expect(Object.keys(scores)).toContain('CPT'); // 或您的实际构念名称
    
    // 3. 准备机器学习数据集
    const mlDataset = prepareMLDataset(scores);
    expect(mlDataset.length).toBeGreaterThan(0);
    
    // 4. 提取特征和目标变量
    // 根据您的实际构念调整特征和目标变量
    const { features, target } = extractFeaturesAndTarget(
      mlDataset, 
      ['CPT', 'RAD', 'CPX'], // 您的特征构念
      'BI' // 您的目标构念
    );
    
    expect(features).toBeDefined();
    expect(target).toBeDefined();
    
    // 5. 分割训练集和测试集
    const { trainFeatures, trainTarget, testFeatures, testTarget } = 
      trainTestSplit(features, target, 0.2);
    
    // 检查分割是否正确
    expect(trainFeatures.length + testFeatures.length).toBe(features.length);
  });
});