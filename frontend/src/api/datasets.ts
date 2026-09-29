import api from './client';

export const getIrisData = (page = 1, pageSize = 50) =>
  api.get(`/datasets/iris?page=${page}&page_size=${pageSize}`);
export const getQuality = () => api.get('/datasets/quality');
export const getDictionary = () => api.get('/datasets/dictionary');
