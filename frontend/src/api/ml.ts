import api from './client';

export const getModelInfo = () => api.get('/ml/model-info');
export const trainModel = () => api.post('/ml/train');
export const getEvaluation = () => api.get('/ml/evaluation');
export const getDecisionBoundary = (kernel: string, featureX = 2, featureY = 3) =>
  api.get(`/ml/decision-boundary?kernel=${kernel}&feature_x=${featureX}&feature_y=${featureY}`);
export const getModelHistory = () => api.get('/ml/model-history');
export const getKernels = () => api.get('/ml/kernels');
