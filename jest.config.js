module.exports = {
    verbose: true,
    testEnvironment: 'node',
    collectCoverage: true,
    collectCoverageFrom: ['**/src/**/*.js'],
    coverageThreshold: {
      global: {
        branches: 80,
        functions: 80,
        lines: 80,
        statements: 80
      }
    }
  };