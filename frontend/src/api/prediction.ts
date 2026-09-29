import api from './client';

export interface PredictRequest {
  sepal_length: number;
  sepal_width: number;
  petal_length: number;
  petal_width: number;
  model?: 'rbf' | 'linear' | 'poly' | 'sigmoid' | 'mlp';
}

export interface SinglePredictionResponse {
  predicted_class_idx: number;
  predicted_species: string;
  probabilities: {
    setosa: number;
    versicolor: number;
    virginica: number;
  };
  confidence: number;
  accuracy: number;
  model: string;
  kernel: string;
  kernel_display: string;
  history_id?: number | string;
}

export interface KernelPrediction {
  kernel: string;
  kernel_display: string;
  predicted_class_idx: number;
  predicted_species: string;
  confidence: number;
  accuracy: number;
  cv_accuracy: number;
  probabilities: {
    setosa: number;
    versicolor: number;
    virginica: number;
  };
}

export interface MultiKernelPredictionResponse {
  features: {
    sepal_length: number;
    sepal_width: number;
    petal_length: number;
    petal_width: number;
  };
  results: Record<string, KernelPrediction>;
  consensus: 'unanimous' | 'majority' | 'divergent';
  consensus_text: string;
  highest_confidence: {
    kernel: string;
    kernel_display: string;
    confidence: number;
    predicted_species: string;
  };
  primary_prediction: string;
  history_id?: number | string;
}

export interface BatchRowResult {
  row_number: number;
  sepal_length: number;
  sepal_width: number;
  petal_length: number;
  petal_width: number;
  rbf_prediction: string;
  rbf_confidence: number;
  rbf_accuracy: number;
  linear_prediction: string;
  linear_confidence: number;
  linear_accuracy: number;
  poly_prediction: string;
  poly_confidence: number;
  poly_accuracy: number;
  sigmoid_prediction?: string;
  sigmoid_confidence?: number;
  sigmoid_accuracy?: number;
  mlp_prediction?: string;
  mlp_confidence?: number;
  mlp_accuracy?: number;
  consensus: string;
}

export interface BatchUploadResponse {
  batch_id: string;
  filename: string;
  total_rows: number;
  warnings: string[];
  results: BatchRowResult[];
  metrics: Record<string, any>;
}

export const predict = (features: PredictRequest) =>
  api.post<SinglePredictionResponse>('/prediction/predict', features);

export const predictAll = (features: PredictRequest) =>
  api.post<MultiKernelPredictionResponse>('/prediction/predict-all', features);

export const uploadBatchFile = (formData: FormData) =>
  api.post<BatchUploadResponse>('/prediction/batch-upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

export const exportBatch = async (data: { batch_id?: string; format: 'xlsx' | 'csv'; rows?: any[] }) => {
  const response = await api.post('/prediction/batch-export', data, {
    responseType: 'blob',
  });
  return response;
};

export const getInsights = (features: PredictRequest) =>
  api.post('/prediction/insights', features);

export const getModelMetrics = () =>
  api.get('/prediction/metrics');

export interface HistoryFilterParams {
  species?: string;
  kernel?: string;
  model?: string;
  start_date?: string;
  end_date?: string;
  search?: string;
  page?: number;
  page_size?: number;
}

export interface PredictionHistoryItem {
  id: number;
  user_id?: number;
  username?: string;
  timestamp: string;
  features: {
    sepal_length: number;
    sepal_width: number;
    petal_length: number;
    petal_width: number;
  };
  predicted_species: string;
  confidence: number;
  model: string;
  kernel: string;
  kernel_display: string;
  probabilities?: {
    setosa: number;
    versicolor: number;
    virginica: number;
  };
}

export interface PaginatedHistoryResponse {
  items: PredictionHistoryItem[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

export const getHistory = (params?: HistoryFilterParams) =>
  api.get<PaginatedHistoryResponse | PredictionHistoryItem[]>('/prediction/history', { params });

export const getAllHistory = (params?: HistoryFilterParams) =>
  api.get<PaginatedHistoryResponse | PredictionHistoryItem[]>('/prediction/history/all', { params });

export const deletePrediction = (predictionId: number | string) =>
  api.delete<{ success: boolean; message: string; prediction_id: number }>(`/prediction/history/${predictionId}`);

export const clearHistory = () =>
  api.delete<{ success: boolean; message: string; count: number }>('/prediction/history');

export const exportHistoryBlob = (format: 'xlsx' | 'csv', params?: HistoryFilterParams) =>
  api.get('/prediction/history/export', {
    params: { ...params, format },
    responseType: 'blob',
  });
